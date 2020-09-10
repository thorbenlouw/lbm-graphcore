

#include <cstdlib>
#include <poplar/IPUModel.hpp>
#include <popops/Reduce.hpp>
#include <popops/codelets.hpp>
#include <iostream>
#include <poplar/Program.hpp>
#include <algorithm>
#include <random>
#include <cxxopts.hpp>
#include <popops/Zero.hpp>

#include "include/GraphcoreUtils.hpp"
#include "include/LbmParams.hpp"
#include "include/LatticeBoltzmannUtils.hpp"
#include "include/StructuredGridUtils.hpp"
#include <poplar/CycleCount.hpp>
#include <poplar/CSRFunctions.hpp>
#include <stdio.h>

using namespace poplar;
using namespace poplar::program;
using namespace popops;
using namespace utils;

auto averageVelocity(Graph &graph, const lbm::Params &params, TensorMap &tensors,
                     const grids::GridPartitioning &workerLevelMappings) -> Program
{

    // As part of collision we already calculated a partialSum (float) and partialCount (unsigned) for each worker
    // which represents the summed normedVelocity and count of cells which are not masked by obstacles. Now we reduce them
    // which represents the summed normedVelocity and count of cells which are not masked by obstacles. Now we reduce them

    // Do multiple reductions in parallel
    std::vector<ComputeSet> reductionComputeSets;
    popops::reduceWithOutput(graph, tensors["perWorkerPartialSums"],
                             tensors["reducedSum"], {0}, {popops::Operation::ADD}, reductionComputeSets,
                             "reducedSums+=perWorkerPartialCounts[i]");

    /* Calculate the average and write it to the relevant place in the array. This happens on every tile,
     * because each tile owns a piece of cells, and only the owner of the piece with the index actually writes */

    const auto numIpus = graph.getTarget().getNumIPUs();
    const auto numTilesPerIpu = graph.getTarget().getNumTiles() / numIpus;

    auto ipuPartitioning = grids::partitionForIpus({params.maxIters, 1}, numIpus, params.maxIters / numIpus);
    assert(ipuPartitioning.has_value());
    auto avVelsTileMapping = grids::toTilePartitions(
        *ipuPartitioning,
        graph.getTarget().getNumIPUs(), 1, 10);

    ComputeSet appendResultCs = graph.addComputeSet("appendReducedSum");
    for (const auto &[target, slice] : avVelsTileMapping)
    {
        auto tile = target.virtualTile(numTilesPerIpu);
        auto appendReducedSumVertex = graph.addVertex(
            appendResultCs,
            "AppendReducedSum",
            {
                {"sumOfVelocities", tensors["reducedSum"]},
                {"indexToWrite", tensors["counter"]},
                {"finals", tensors["av_vel"].slice(
                                                slice.rows().from(),
                                                slice.rows().to(),
                                                0)
                               .flatten()},
            });
        graph.setInitialValue(appendReducedSumVertex["myStartIndex"], slice.rows().from());
        graph.setInitialValue(appendReducedSumVertex["myEndIndex"], slice.rows().to() - 1);
        graph.setTileMapping(tensors["av_vel"].slice(slice.rows().from(),
                                                     slice.rows().to(), 0),
                             tile);
        graph.setCycleEstimate(appendReducedSumVertex, 4);
        graph.setTileMapping(appendReducedSumVertex, tile);
    }

    ComputeSet incrementCs = graph.addComputeSet("increment");

    auto incrementVertex = graph.addVertex(incrementCs,
                                           "IncrementIndex", // Create a vertex of this
                                           {
                                               {"index", tensors["counter"]} // Connect input 'b' of the
                                           });
    graph.setCycleEstimate(incrementVertex, 13);
    graph.setTileMapping(incrementVertex, numTilesPerIpu - 1);

    Sequence seq;
    for (const auto &cs : reductionComputeSets)
    {
        seq.add(Execute(cs));
    }
    seq.add(Execute(appendResultCs));
    seq.add(Execute(incrementCs));
    return std::move(seq);
}

auto totalDensity(Graph &graph, const lbm::Params &params, TensorMap &tensors) -> Sequence
{
    auto s = Sequence();
    tensors["totalDensity"] = popops::reduce(graph, tensors["cells"], FLOAT, {2, 1, 0},
                                             {popops::Operation::ADD, false}, s);
    return s;
}

auto collision(Graph &graph, const lbm::Params &params, TensorMap &tensors,
               const grids::GridPartitioning &mappings, const unsigned numWorkersPerTile) -> Program
{
    const auto numTilesPerIpu = graph.getTarget().getNumTiles() / graph.getTarget().getNumIPUs();

    ComputeSet collisionCs = graph.addComputeSet("collision");

    for (const auto &[target, slice] : mappings)
    {
        auto tile = target.virtualTile(numTilesPerIpu);
        auto numCellsForThisWorker = slice.width() * slice.height();

        auto v = graph.addVertex(collisionCs,
                                 "CollisionVertex",
                                 {{"in", applySlice(tensors["tmp_cells"], slice).flatten()},
                                  {"out", applySlice(tensors["cells"], slice).flatten()},
                                  {"obstacles", applySlice(tensors["obstacles"], slice).flatten()},
                                  {"normedVelocityPartial", tensors["perWorkerPartialSums"][tile * numWorkersPerTile +
                                                                                            target.worker()]}});
        graph.setInitialValue(v["width"], slice.width());
        graph.setInitialValue(v["height"], slice.height());
        graph.setInitialValue(v["omega"], params.omega);

        graph.setCycleEstimate(v, numCellsForThisWorker);
        graph.setTileMapping(v, tile);
    }

    return Execute(collisionCs);
}

auto propagate(Graph &graph,
               const lbm::Params &params, TensorMap &tensors,
               const grids::GridPartitioning &mappings, const unsigned numWorkersPerTile) -> Program
{
    const auto numTilesPerIpu = graph.getTarget().getNumTiles() / graph.getTarget().getNumIPUs();

    ComputeSet in2outCs = graph.addComputeSet("in2out");
    ComputeSet out2inCs = graph.addComputeSet("out2in");
    auto cells = tensors["cells"];

    auto fullSize = grids::Size2D(params.ny, params.nx);

    for (const auto &[target, slice] : mappings)
    {

        auto tile = target.virtualTile(numTilesPerIpu);
        auto numCellsForThisWorker = slice.width() * slice.height();

        auto halos = grids::Halos::forSliceWithWraparound(slice, fullSize);

        auto inWithHalos = stitchHalos(applySlice(cells, *halos.topLeft), applySlice(cells, *halos.top),
                                       applySlice(cells, *halos.topRight),
                                       applySlice(cells, *halos.left), applySlice(cells, slice),
                                       applySlice(cells, *halos.right),
                                       applySlice(cells, *halos.bottomLeft), applySlice(cells, *halos.bottom),
                                       applySlice(cells, *halos.bottomRight))
                               .flatten();
        auto v = graph.addVertex(in2outCs,
                                 "PropagateVertexAoS",
                                 {
                                     {"in", inWithHalos},
                                     {"out", applySlice(tensors["tmp_cells"], slice).flatten()},
                                 });
        graph.setInitialValue(v["width"], slice.width());
        graph.setInitialValue(v["height"], slice.height());
        graph.setCycleEstimate(v, numCellsForThisWorker);
        graph.setTileMapping(v, tile);

        inWithHalos = stitchHalos(applySlice(tensors["tmp_cells"], *halos.topLeft),
                                  applySlice(tensors["tmp_cells"], *halos.top),
                                  applySlice(tensors["tmp_cells"], *halos.topRight),
                                  applySlice(tensors["tmp_cells"], *halos.left),
                                  applySlice(tensors["tmp_cells"], slice),
                                  applySlice(tensors["tmp_cells"], *halos.right),
                                  applySlice(tensors["tmp_cells"], *halos.bottomLeft),
                                  applySlice(tensors["tmp_cells"], *halos.bottom),
                                  applySlice(tensors["tmp_cells"], *halos.bottomRight))
                          .flatten();
        v = graph.addVertex(out2inCs,
                            "PropagateVertexAoS",
                            {{"in", inWithHalos},
                             {"out", applySlice(tensors["cells"], slice).flatten()}});
        graph.setInitialValue(v["width"], slice.width());
        graph.setInitialValue(v["height"], slice.height());
        graph.setCycleEstimate(v, numCellsForThisWorker);
        graph.setTileMapping(v, tile);
    }

    return Sequence{Execute(in2outCs), Execute(out2inCs)};
}

auto combo(Graph &graph,
           const lbm::Params &params, TensorMap &tensors,
           const grids::GridPartitioning &mappings, const unsigned numWorkersPerTile) -> Program
{
    const auto numTilesPerIpu = graph.getTarget().getNumTiles() / graph.getTarget().getNumIPUs();

    ComputeSet in2outCs = graph.addComputeSet("in2out");
    ComputeSet out2inCs = graph.addComputeSet("out2in");
    auto cells = tensors["cells"];

    auto fullSize = grids::Size2D(params.ny, params.nx);

    auto i = 0;
    for (const auto &[target, slice] : mappings)
    {
        i++;
        // Accelerate if vertex is responsible for row  params.ny - 2
        auto isAcceleratingVertex = (slice.rows().from() <= params.ny - 2) && (slice.rows().to() > params.ny - 2);
        auto rowToAccelerate = (isAcceleratingVertex) ? params.ny - 2 - slice.rows().from() : 0;

        auto tile = target.virtualTile(numTilesPerIpu);
        auto numCellsForThisWorker = slice.width() * slice.height();

        auto halos = grids::Halos::forSliceWithWraparound(slice, fullSize);

        auto inWithHalos = stitchHalos(applySlice(cells, *halos.topLeft), applySlice(cells, *halos.top),
                                       applySlice(cells, *halos.topRight),
                                       applySlice(cells, *halos.left), applySlice(cells, slice),
                                       applySlice(cells, *halos.right),
                                       applySlice(cells, *halos.bottomLeft), applySlice(cells, *halos.bottom),
                                       applySlice(cells, *halos.bottomRight))
                               .flatten();
        auto v = graph.addVertex(in2outCs,
                                 "LbmVertex",
                                 {{"in", inWithHalos},
                                  {"out", applySlice(tensors["tmp_cells"], slice).flatten()},
                                  {"obstacles", applySlice(tensors["obstacles"], slice).flatten()},
                                  {"normedVelocityPartial", tensors["perWorkerPartialSums"][tile * numWorkersPerTile +
                                                                                            target.worker()]}});
        graph.setInitialValue(v["width"], slice.width());
        graph.setInitialValue(v["height"], slice.height());
        graph.setInitialValue(v["isAcceleratingVertex"], isAcceleratingVertex);
        graph.setInitialValue(v["rowToAccelerate"], rowToAccelerate);
        graph.setInitialValue(v["omega"], params.omega);
        graph.setInitialValue(v["oneMinusOmega"], 1.f - params.omega);
        graph.setInitialValue(v["w1"], params.density * params.accel / 9.0f);
        graph.setInitialValue(v["w2"], params.density * params.accel / 36.0f);
        graph.setCycleEstimate(v, numCellsForThisWorker);
        graph.setTileMapping(v, tile);

        inWithHalos = stitchHalos(applySlice(tensors["tmp_cells"], *halos.topLeft),
                                  applySlice(tensors["tmp_cells"], *halos.top),
                                  applySlice(tensors["tmp_cells"], *halos.topRight),
                                  applySlice(tensors["tmp_cells"], *halos.left),
                                  applySlice(tensors["tmp_cells"], slice),
                                  applySlice(tensors["tmp_cells"], *halos.right),
                                  applySlice(tensors["tmp_cells"], *halos.bottomLeft),
                                  applySlice(tensors["tmp_cells"], *halos.bottom),
                                  applySlice(tensors["tmp_cells"], *halos.bottomRight))
                          .flatten();
        v = graph.addVertex(out2inCs,
                            "LbmVertex",
                            {{"in", inWithHalos},
                             {"out", applySlice(tensors["cells"], slice).flatten()},
                             {"obstacles", applySlice(tensors["obstacles"], slice).flatten()},
                             {"normedVelocityPartial", tensors["perWorkerPartialSums"][tile * numWorkersPerTile +
                                                                                       target.worker()]}});
        graph.setInitialValue(v["width"], slice.width());
        graph.setInitialValue(v["height"], slice.height());
        graph.setInitialValue(v["isAcceleratingVertex"], isAcceleratingVertex);
        graph.setInitialValue(v["rowToAccelerate"], rowToAccelerate);
        graph.setInitialValue(v["omega"], params.omega);
        graph.setInitialValue(v["oneMinusOmega"], 1.f - params.omega);
        graph.setInitialValue(v["w1"], params.density * params.accel / 9.0f);
        graph.setInitialValue(v["w2"], params.density * params.accel / 36.0f);

        graph.setCycleEstimate(v, numCellsForThisWorker);
        graph.setTileMapping(v, tile);
    }
    printf("Add %d LbmVertex instances\n", i * 2);

    const auto doAverage = averageVelocity(graph, params, tensors, mappings);

    return Sequence{Execute(in2outCs), doAverage, Execute(out2inCs), doAverage};
}

auto accelerate_flow(Graph &graph, const lbm::Params &params, TensorMap &tensors,
                     const unsigned numWorkersPerTile) -> Program
{

    const auto numTilesPerIpu = graph.getTarget().getNumTiles() / graph.getTarget().getNumIPUs();

    ComputeSet accelerateCs = graph.addComputeSet("accelerate");

    auto cells = tensors["cells"];
    auto obstacles = tensors["obstacles"];
    //    assert(cells.dim(0) > 1);
    auto cellsSecondRowFromTop = cells.slice(cells.dim(0) - 2, cells.dim(0) - 1, 0);
    auto obstaclesSecondRowFromTop = obstacles.slice(cells.dim(0) - 2, cells.dim(0) - 1, 0);

    const auto ipuLevelMapping = grids::partitionForIpus({1, params.nx}, graph.getTarget().getNumIPUs(),
                                                         params.nx);

    assert(ipuLevelMapping.has_value());
    auto tileGranularityMappings = grids::newTilePartitions(
        *ipuLevelMapping,
        numTilesPerIpu);
    auto workerGranularityMappings = grids::toWorkerPartitions(
        tileGranularityMappings,
        numWorkersPerTile);

    for (const auto &[target, slice] : workerGranularityMappings)
    {

        auto tile = target.virtualTile(numTilesPerIpu);

        auto numCellsForThisWorker = slice.width() * slice.height();

        auto v = graph.addVertex(accelerateCs,
                                 "AccelerateFlowVertex",
                                 {{"cellsInSecondRow", applySlice(cellsSecondRowFromTop, slice).flatten()},
                                  {"obstaclesInSecondRow", applySlice(obstaclesSecondRowFromTop,
                                                                      slice)
                                                               .flatten()}});
        graph.setInitialValue(v["width"], numCellsForThisWorker);
        graph.setInitialValue(v["w1"], params.density * params.accel / 9.0f);
        graph.setInitialValue(v["w2"], params.density * params.accel / 36.0f);
        graph.setCycleEstimate(v, numCellsForThisWorker);
        graph.setTileMapping(v, tile);
    }

    return Execute(accelerateCs);
}

auto timestep(Graph &graph, const lbm::Params &params, TensorMap &tensors,
              const grids::GridPartitioning &mappings, const unsigned numWorkersPerTile) -> Program
{
    return Sequence{

        combo(graph, params, tensors, mappings, numWorkersPerTile),

    };
}

auto main(int argc, char *argv[]) -> int
{
    std::string obstaclesFileArg, paramsFileArg;
    unsigned numIpusArg = 1u;
    bool useModel = false;
    bool debug = false;
    cxxopts::Options options(argv[0], " - Runs a Lattice Boltzmann graph on the IPU or IPU Model");
    options.add_options()("d,debug", "Capture profiling")("m,use-ipu-model", "Use  IPU model instead of real device")("n,num-ipus", "number of ipus to use", cxxopts::value<unsigned>(numIpusArg)->default_value("1"))("params", "filename of parameters file", cxxopts::value<std::string>(paramsFileArg))("obstacles", "filename of obstacles file", cxxopts::value<std::string>(obstaclesFileArg));
    try
    {
        auto opts = options.parse(argc, argv);
        debug = opts["d"].as<bool>();
        useModel = opts["m"].as<bool>();
        if (opts.count("n") + opts.count("params") + opts.count("obstacles") < 3)
        {
            std::cerr << options.help() << std::endl;
            return EXIT_FAILURE;
        }
    }
    catch (cxxopts::OptionParseException &)
    {
        std::cerr << options.help() << std::endl;
        return EXIT_FAILURE;
    }
    if (debug)
    {
        std::cout << "Capturing profile information during this run." << std::endl;
    };

    auto params = lbm::Params::fromFile(paramsFileArg);
    if (!params.has_value())
    {
        std::cerr << "Could not parse parameters file. Aborting" << std::endl;
        return EXIT_FAILURE;
    }

    auto obstacles = lbm::Obstacles::fromFile(params->nx, params->ny, obstaclesFileArg);
    if (!obstacles.has_value())
    {
        std::cerr << "Could not parse obstacles file" << std::endl;
        return EXIT_FAILURE;
    }
    auto numNonObstacleCells = [&]() -> long {
        long total = 0l;
        for (auto y = 0u; y < obstacles->ny; y++)
        {
            for (auto x = 0u; x < obstacles->nx; x++)
            {
                total += !obstacles->at(x, y);
            }
        }
        return total;
    }();

    auto device = useModel ? utils::getIpuModel(numIpusArg) : utils::getIpuDevice(numIpusArg);
    if (!device.has_value())
    {
        return EXIT_FAILURE;
    }

    auto target = device->getTarget();
    auto numTilesPerIpu = target.getNumTiles() / target.getNumIPUs();
    auto numWorkersPerTile = target.getNumWorkerContexts();
    auto numIpus = target.getNumIPUs();

    const auto ipuLevelMapping = grids::partitionForIpus({params->ny, params->nx}, numIpus, 2000 * 1000);
    if (!ipuLevelMapping.has_value())
    {
        std::cerr << "Couldn't find a way to partition the input parameter space over the given number of IPUs"
                  << std::endl;
        return EXIT_FAILURE;
    }

    grids::serializeToJson(*ipuLevelMapping, "ipu-mapping.json");

    const auto tileGranularityMappings = grids::newTilePartitions(*ipuLevelMapping,
                                                                  numTilesPerIpu);

    //    const auto tileGranularityMappings = grids::lbm1024x1024TilePartitions(*ipuLevelMapping,
    //                                                                    numTilesPerIpu);
    const auto workerGranularityMappings = grids::toWorkerPartitions(
        tileGranularityMappings, numWorkersPerTile);

    grids::serializeToJson(workerGranularityMappings, "partitioning.json");

    auto tensors = std::map<std::string, Tensor>{};
    auto programs = std::vector<Program>{};

    Graph graph(target);

    timedStep("Building computational graph",
              [&]() {
        popops::addCodelets(graph);

        graph.addCodelets("codelets/D2Q9Codelets.cpp", CodeletFileType::Auto, debug ? "" : "-O3");

        tensors["av_vel"] = graph.addVariable(FLOAT, {params->maxIters, 1},
                                              "av_vel");
        tensors["cells"] = graph.addVariable(FLOAT, {params->ny, params->nx, lbm::NumSpeeds},
                                             "cells");
        tensors["tmp_cells"] = graph.addVariable(FLOAT, {params->ny, params->nx, lbm::NumSpeeds},
                                                 "tmp_cells");
        tensors["obstacles"] = graph.addVariable(BOOL, {params->ny, params->nx}, "obstacles");
        tensors["perWorkerPartialSums"] = graph.addVariable(FLOAT,
                                                            {numWorkersPerTile * numTilesPerIpu * numIpus},
                                                            poplar::VariableMappingMethod::LINEAR,
                                                            "perWorkerPartialSums");

        tensors["reducedSum"] = graph.addVariable(FLOAT, {}, "reducedSum");
        graph.setInitialValue(tensors["reducedSum"], 0.f);
        graph.setTileMapping(tensors["reducedSum"], 0);

        mapCellsToTiles(graph, tensors["cells"], tileGranularityMappings);
        mapCellsToTiles(graph, tensors["tmp_cells"], tileGranularityMappings);
        mapCellsToTiles(graph, tensors["obstacles"], tileGranularityMappings);

        tensors["counter"] = graph.addVariable(UNSIGNED_INT, {}, "counter");
        graph.setTileMapping(tensors["counter"], numTilesPerIpu - 1);
        graph.setInitialValue(tensors["counter"], 0);

        auto outStreamAveVelocities = graph.addDeviceToHostFIFO("<<av_vel", FLOAT, params->maxIters);
        auto outStreamFinalCells = graph.addDeviceToHostFIFO("<<cells", FLOAT,
                                                             lbm::NumSpeeds * params->nx *
                                                                 params->ny);
        auto inStreamCells = graph.addHostToDeviceFIFO(">>cells", FLOAT,
                                                       lbm::NumSpeeds * params->nx * params->ny);
        auto inStreamObstacles = graph.addHostToDeviceFIFO(">>obstacles", BOOL,
                                                           params->nx * params->ny);

        auto streamBackToHostProg = Sequence(
            Copy(tensors["cells"], outStreamFinalCells),
            Copy(tensors["av_vel"], outStreamAveVelocities));

        averageVelocity(graph, *params, tensors, workerGranularityMappings);
        auto firstAccelCs = graph.addComputeSet("firstAccel");

        auto firstAccelVertex = graph.addVertex(
            firstAccelCs,
            "FirstAccelVertex",
            {
                {"cellsVec", tensors["cells"].flatten()},
                {"obstaclesVec", tensors["obstacles"].flatten()},
            });
        graph.setInitialValue(firstAccelVertex["ny"], params->ny);
        graph.setInitialValue(firstAccelVertex["nx"], params->nx);
        graph.setInitialValue(firstAccelVertex["density"], params->density);
        graph.setInitialValue(firstAccelVertex["accel"], params->accel);
        graph.setCycleEstimate(firstAccelVertex, 4);
        graph.setTileMapping(firstAccelVertex, 1);

        auto cs = graph.addComputeSet("lastChance");

        auto lastChanceVertex = graph.addVertex(
            cs,
            "LastHopeVertex",
            {
                {"cellsVec", tensors["cells"].flatten()},
                {"tmp_cellsVec", tensors["tmp_cells"].flatten()},
                {"obstaclesVec", tensors["obstacles"].flatten()},
                {"av_velsVec", tensors["av_vel"].flatten()}
            });
        graph.setInitialValue(lastChanceVertex["ny"], params->ny);
        graph.setInitialValue(lastChanceVertex["nx"], params->nx);
        graph.setInitialValue(lastChanceVertex["maxIters"], params->maxIters);
        graph.setInitialValue(lastChanceVertex["omega"], params->omega);
        graph.setInitialValue(lastChanceVertex["one_minus_omega"], 1.f - params->omega);
        graph.setInitialValue(lastChanceVertex["density"], params->density);
        graph.setInitialValue(lastChanceVertex["accel"], params->accel);
        graph.setInitialValue(lastChanceVertex["total_free_cells"], numNonObstacleCells);
        graph.setCycleEstimate(lastChanceVertex, 4);
        graph.setTileMapping(lastChanceVertex, 0);

        auto prog = Sequence();
         poplar::setFloatingPointBehaviour(graph, prog, {true, true, true, false, true},
                                                    "no stochastic rounding");
        prog.add(Execute(firstAccelCs));
        prog.add(Execute(cs));

            //   accelerate_flow(graph, *params, tensors, numWorkersPerTile),
            //   Repeat(params->maxIters / 2, Sequence{
            //                          totalDensity(graph, *params, tensors),
            //                          PrintTensor("totalDensity (before): ", tensors["totalDensity"]),
            //                          PrintTensor("Obstacles: ", tensors["obstacles"]),
            //                          PrintTensor("Cells (before): ", tensors["cells"]),
            //   timestep(graph, *params, tensors, workerGranularityMappings,
            //            numWorkersPerTile),
            //                          PrintTensor("Cells: (after) ", tensors["cells"]),

            //                          totalDensity(graph, *params, tensors),
            //                          PrintTensor("totalDensity (after): ", tensors["totalDensity"])
        
                 

                  auto copyCellsAndObstaclesToDevice = Sequence();
                  popops::zero(graph, tensors["av_vel"], copyCellsAndObstaclesToDevice, "av_vels=0");
                  popops::zero(graph, tensors["cells"], copyCellsAndObstaclesToDevice, "cells=0");
                  popops::zero(graph, tensors["tmp_cells"], copyCellsAndObstaclesToDevice, "tmp_cells=0");
                  copyCellsAndObstaclesToDevice.add(Copy(inStreamCells, tensors["cells"]));
                  copyCellsAndObstaclesToDevice.add(Copy(inStreamObstacles, tensors["obstacles"]));

                  auto timing = poplar::cycleCount(graph,
                                                   prog,
                                                   0, "timer");


                  graph.createHostRead("readTimer", timing, true);
                  programs.push_back(copyCellsAndObstaclesToDevice);
                  programs.push_back(prog);
                  programs.push_back(streamBackToHostProg);

                  if (auto dumpGraphVisualisations =
                              std::getenv("DUMP_GRAPH_VIZ") != nullptr;  dumpGraphVisualisations) {
        ofstream vertexGraph;
        vertexGraph.open("vertexgraph.dot");
        graph.outputVertexGraph(vertexGraph,
                                programs);
        vertexGraph.close();

        ofstream computeGraph;
        computeGraph.open("computegraph.dot");
        graph.outputComputeGraph(computeGraph,
                                 programs);
        computeGraph.close();
                  }
});

timedStep("Serializing graph for analysis", [&]() -> void {
    serializeGraph(graph);
});

double total_compute_time = 0.0;
auto cells = lbm::Cells(params->nx, params->ny);
cells.initialise(*params);
std::cout << "HOST total density: " << cells.totalDensity() << std::endl;

auto av_vels = std::vector<float>(params->maxIters, 0.0f);

auto engine = Engine(graph, programs,
                     debug ? utils::POPLAR_ENGINE_OPTIONS_DEBUG : utils::POPLAR_ENGINE_OPTIONS_NODEBUG);

engine.connectStream("<<av_vel", av_vels.data());
engine.connectStream("<<cells", cells.data.data());
engine.connectStream(">>cells", cells.data.data());
engine.connectStream(">>obstacles", obstacles->getData());

utils::timedStep("Loading graph to device", [&]() {
    engine.load(*device);
});

utils::timedStep("Running copy to device step", [&]() {
    engine.run(0);
});

total_compute_time += utils::timedStep("Running LBM", [&]() {
    engine.run(1);
});

utils::timedStep("Running copy to host step", [&]() {
    engine.run(2);
});

//    for (auto jj = 0u; jj < params->ny; jj++) {
//        for (auto ii = 0u; ii < params->nx; ii++) {
//            for (auto kk = 0u; kk < 9; kk++) {
//                printf("%.9f ", cells.data[9 * (ii + jj * params->nx) + kk]);
//            }
//            printf("\n");
//        }
//        printf("\n");
//    }

int numErrs = 0;
for (auto jj = 0u; jj < params->ny; jj++)
{
    for (auto ii = 0u; ii < params->nx; ii++)
    {
        for (auto k = 0u; k < 9; k++)
        {
            if (std::isnan(cells.data[9 * (ii + jj * params->nx) + k]))
            {
                if (numErrs < 100)
                {
                    printf("%d %d %d is %.12E\n", jj, ii, k, cells.data[9 * (ii + jj * params->nx) + k]);
                }
                numErrs++;
            }
        }
    }
}
printf("%d Errors\n", numErrs);

utils::timedStep("Writing output files ", [&]() {
    // for (auto i = 0u; i < av_vels.size(); i++)
    // {
    //     //            std::cout << av_vels[i] << std::endl;
    //     av_vels[i] = av_vels[i] / numNonObstacleCells;
    // }
    lbm::writeAverageVelocities("av_vels.dat", av_vels);
    lbm::writeResults("final_state.dat", *params, *obstacles, cells);
});

if (debug)
{
    utils::timedStep("Capturing profiling info", [&]() {
        utils::captureProfileInfo(engine);
    });

    engine.printProfileSummary(std::cout,
                               OptionFlags{{"showExecutionSteps", "false"}});
}

std::cout << "==done==" << std::endl;
std::cout << "Total compute time was \t" << std::right << std::setw(12) << std::setprecision(5)
          << total_compute_time
          << "s" << std::endl;

std::cout << "Reynolds number:  \t" << std::right << std::setw(12) << std::setprecision(12)
          << std::scientific
          << lbm::reynoldsNumber(*params, av_vels[params->maxIters - 1]) << std::endl;

std::cout << "HOST total density: " << cells.totalDensity() << std::endl;

cout << "Now doing 5 runs and averaging IPU-reported timing:" << std::endl;
unsigned long ipuTimer;
double clockCycles = 0.;
for (auto run = 0u; run < 5u; run++)
{
    engine.run(2);
    engine.readTensor("readTimer", &ipuTimer);
    clockCycles += ipuTimer;
}
double clockFreq = device->getTarget().getTileClockFrequency();
std::cout << "IPU reports " << std::fixed << clockFreq * 1e-6 << "MHz clock frequency" << std::endl;
std::cout << "Average IPU timing for program is: " << std::fixed << std::setprecision(5) << std::setw(12)
          << clockCycles / 5.0 / clockFreq << "s" << std::endl;

std::cout << "==done==" << std::endl;

return EXIT_SUCCESS;
}
