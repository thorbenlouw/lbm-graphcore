

#include <cstdlib>
#include <poplar/IPUModel.hpp>
#include <popops/Reduce.hpp>
#include <popops/codelets.hpp>
#include <iostream>
#include <poplar/Program.hpp>
#include <algorithm>
#include <random>
#include <cxxopts.hpp>

#include "include/GraphcoreUtils.hpp"
#include "include/LbmParams.hpp"
#include "include/LatticeBoltzmannUtils.hpp"
#include "include/StructuredGridUtils.hpp"
#include <poplar/CycleCount.hpp>


using namespace poplar;
using namespace poplar::program;
using namespace popops;
using namespace utils;

auto
averageVelocity(Graph &graph, const lbm::Params &params, TensorMap &tensors,
                const grids::GridPartitioning &workerLevelMappings) -> Program {

    // As part of collision we already calculated a partialSum (float) and partialCount (unsigned) for each worker
    // which represents the summed normedVelocity and count of cells which are not masked by obstacles. Now we reduce them

    // Do multiple reductions in parallel
    std::vector <ComputeSet> reductionComputeSets;
    popops::reduceWithOutput(graph, tensors["perWorkerPartialCounts"],
                             tensors["reducedCount"], {0}, {popops::Operation::ADD}, reductionComputeSets,
                             "reducedCount+=perWorkerPartialCounts[i]");
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
            graph.getTarget().getNumIPUs()
    );

    ComputeSet appendResultCs = graph.addComputeSet("appendReducedSum");
    for (const auto &[target, slice] : avVelsTileMapping) {
        auto tile = target.virtualTile(numTilesPerIpu);

        auto appendReducedSumVertex = graph.addVertex(
                appendResultCs,
                "AppendReducedSum",
                {
                        {"total",        tensors["reducedSum"]},
                        {"count",        tensors["reducedCount"]},
                        {"indexToWrite", tensors["counter"]},
                        {"myStartIndex", slice.rows().from()},
                        {"myEndIndex",   slice.rows().to() - 1},
                        {"finals",       tensors["av_vel"].slice(
                                slice.rows().from(),
                                slice.rows().to(),
                                0).flatten()},
                }
        );
        graph.setTileMapping(tensors["av_vel"].slice(slice.rows().from(),
                                                     slice.rows().to(),
                                                     0), tile);
        graph.setCycleEstimate(appendReducedSumVertex, 16);
        graph.setTileMapping(appendReducedSumVertex, tile);

    }


    ComputeSet incrementCs = graph.addComputeSet("increment");

    auto incrementVertex = graph.addVertex(incrementCs,
                                           "IncrementIndex", // Create a vertex of this
                                           {{"index", tensors["counter"]}   // Connect input 'b' of the
                                           });
    graph.setCycleEstimate(incrementVertex, 13);
    graph.setTileMapping(incrementVertex, numTilesPerIpu - 1);


    Sequence seq;
    for (const auto &cs : reductionComputeSets) {
        seq.add(Execute(cs));
    }
    seq.add(Execute(appendResultCs));
    seq.add(Execute(incrementCs));
    return std::move(seq);
}

auto
collision(Graph &graph, const lbm::Params &params, TensorMap &tensors,
          const grids::GridPartitioning &mappings) -> Program {
    const auto numTilesPerIpu = graph.getTarget().getNumTiles() / graph.getTarget().getNumIPUs();
    const auto numWorkersPerTile = graph.getTarget().getNumWorkerContexts();

    ComputeSet collisionCs = graph.addComputeSet("collision");

    for (const auto &[target, slice] : mappings) {
        auto tile = target.virtualTile(numTilesPerIpu);
        auto numCellsForThisWorker = slice.width() * slice.height();

        auto v = graph.addVertex(collisionCs,
                                 "CollisionVertex",
                                 {
                                         {"in",                    applySlice(tensors["tmp_cells"], slice).flatten()},
                                         {"out",                   applySlice(tensors["cells"], slice).flatten()},
                                         {"numRows",               slice.height()},
                                         {"numCols",               slice.width()},
                                         {"omega",                 params.omega},
                                         {"obstacles",             applySlice(tensors["obstacles"], slice).flatten()},
                                         {"normedVelocityPartial", tensors["perWorkerPartialSums"][
                                                                           tile * numWorkersPerTile +
                                                                           target.worker()]},
                                         {"countPartial",          tensors["perWorkerPartialCounts"][
                                                                           tile * numWorkersPerTile +
                                                                           target.worker()]}
                                 }
        );
        graph.setCycleEstimate(v, numCellsForThisWorker);
        graph.setTileMapping(v, tile);
    }

    return
            Execute(collisionCs);
}

auto propagate(Graph &graph,
               const lbm::Params &params, TensorMap &tensors,
               const grids::GridPartitioning &mappings) -> Program {
    const auto numTilesPerIpu = graph.getTarget().getNumTiles() / graph.getTarget().getNumIPUs();

    ComputeSet propagateCs = graph.addComputeSet("propagate");
    auto cells = tensors["cells"];

    auto fullSize = grids::Size2D(params.ny, params.nx);
    for (const auto &[target, slice] : mappings) {
        auto tile = target.virtualTile(numTilesPerIpu);
        auto numCellsForThisWorker = slice.width() * slice.height();


        auto halos = grids::Halos::forSliceWithWraparound(slice, fullSize);


        auto v = graph.addVertex(propagateCs,
                                 "PropagateVertex",
                                 {
                                         {"in",              applySlice(cells, slice).flatten()},
                                         {"out",             applySlice(tensors["tmp_cells"], slice).flatten()},
                                         {"numRows",         slice.height()},
                                         {"numCols",         slice.width()},
                                         {"haloTop",         applySlice(cells, *halos.top).flatten()},
                                         {"haloBottom",      applySlice(cells, *halos.bottom).flatten()},
                                         {"haloLeft",        applySlice(cells, *halos.left).flatten()},
                                         {"haloRight",       applySlice(cells, *halos.right).flatten()},
                                         {"haloTopLeft",     applySlice(cells,
                                                                        *halos.topLeft).flatten()[lbm::SpeedIndexes::SouthEast]}, // flipped directions!
                                         {"haloTopRight",    applySlice(cells,
                                                                        *halos.topRight).flatten()[lbm::SpeedIndexes::SouthWest]},// flipped directions!
                                         {"haloBottomLeft",  applySlice(cells,
                                                                        *halos.bottomLeft).flatten()[lbm::SpeedIndexes::NorthEast]},// flipped directions!
                                         {"haloBottomRight", applySlice(cells,
                                                                        *halos.bottomRight).flatten()[lbm::SpeedIndexes::NorthWest]},// flipped directions!
                                 });
        graph.setCycleEstimate(v, numCellsForThisWorker);
        graph.setTileMapping(v, tile);
    }

    return Execute(propagateCs);
}

auto accelerate_flow(Graph &graph, const lbm::Params &params, TensorMap &tensors) -> Program {

    const auto numTilesPerIpu = graph.getTarget().getNumTiles() / graph.getTarget().getNumIPUs();
    const auto numWorkers = graph.getTarget().getNumWorkerContexts();


    ComputeSet accelerateCs = graph.addComputeSet("accelerate");

    auto cells = tensors["cells"];
    auto obstacles = tensors["obstacles"];
    assert(cells.dim(0) > 1);
    auto cellsSecondRowFromTop = cells.slice(cells.dim(0) - 2, cells.dim(0) - 1, 0);
    auto obstaclesSecondRowFromTop = obstacles.slice(cells.dim(0) - 2, cells.dim(0) - 1, 0);

    const auto ipuLevelMapping = grids::partitionForIpus({1, params.nx}, graph.getTarget().getNumIPUs(), params.nx);

    assert(ipuLevelMapping.has_value());
    auto tileGranularityMappings = grids::toTilePartitions(
            *ipuLevelMapping,
            numTilesPerIpu, 1, 48
    );
    auto workerGranularityMappings = grids::toWorkerPartitions(
            tileGranularityMappings,
            numWorkers
    );

    for (const auto &[target, slice] : workerGranularityMappings) {

        auto tile = target.virtualTile(numTilesPerIpu);

        auto numCellsForThisWorker = slice.width() * slice.height();

        auto v = graph.addVertex(accelerateCs,
                                 "AccelerateFlowVertex",
                                 {{"cellsInSecondRow",     applySlice(cellsSecondRowFromTop, slice).flatten()},
                                  {"obstaclesInSecondRow", applySlice(obstaclesSecondRowFromTop, slice).flatten()},
                                  {"partitionWidth",       numCellsForThisWorker},
                                  {"density",              params.density},
                                  {"accel",                params.accel}});
        graph.setCycleEstimate(v, numCellsForThisWorker);
        graph.setTileMapping(v, tile);
    }

    return Execute(accelerateCs);
}


auto
timestep(Graph &graph, const lbm::Params &params, TensorMap &tensors,
         const grids::GridPartitioning &mappings) -> Program {
    return Sequence{
//            accelerate_flow(graph, params, tensors),
            propagate(graph, params, tensors, mappings),
//            collision(graph, params, tensors, mappings)
    };
}

auto main(int argc, char *argv[]) -> int {
    std::string outputFilename, paramsFileArg;
    unsigned numIpusArg = 1u;
    bool debug = false;

    cxxopts::Options options(argv[0], " - Runs a Lattice Boltzmann graph on the IPU or IPU Model");
    options.add_options()
            ("d,debug", "Capture profiling")
            ("n,num-ipus", "number of ipus to use", cxxopts::value<unsigned>(numIpusArg)->default_value("1"))
            ("params", "filename of parameters file", cxxopts::value<std::string>(paramsFileArg))
            ("o,output", "filename of compiled graph", cxxopts::value<std::string>(outputFilename));
    try {
        auto opts = options.parse(argc, argv);
        debug = opts["debug"].as<bool>();
        if (opts.count("n") + opts.count("params") + opts.count("num-ipus") + opts.count("o") < 4) {
            std::cerr << options.help() << std::endl;
            return EXIT_FAILURE;
        }
    } catch (cxxopts::OptionParseException &) {
        std::cerr << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    if (debug) {
        std::cout << "Configuring executable to capture profile information" << std::endl;
    }

    auto params = lbm::Params::fromFile(paramsFileArg);
    if (!params.has_value()) {
        std::cerr << "Could not parse parameters file. Aborting" << std::endl;
        return EXIT_FAILURE;
    }

    auto device = std::optional < Device > {};
    if (debug) {
        device = utils::getIpuDevice(numIpusArg);
        if (!device.has_value()) {
            return EXIT_FAILURE;
        }
    }
//
//    auto target = debug ? device->getTarget()
//                        : Target::createIPUTarget(numIpusArg,
//                                                  "ipu1",
//                                                  debug ? utils::POPLAR_ENGINE_OPTIONS_DEBUG
//                                                        : utils::POPLAR_ENGINE_OPTIONS_NODEBUG);

    auto  target =device->getTarget();
    auto numTilesPerIpu = target.getNumTiles() / target.getNumIPUs();
    auto numWorkers = target.getNumWorkerContexts();
    auto numIpus = target.getNumIPUs();

    const auto ipuLevelMapping = grids::partitionForIpus({params->ny, params->nx}, numIpus, 2000 * 1000);
    if (!ipuLevelMapping.has_value()) {
        std::cerr << "Couldn't find a way to partition the input parameter space over the given number of IPUs"
                  << std::endl;
        return EXIT_FAILURE;
    }
    const auto tileGranularityMappings = grids::toTilePartitions(*ipuLevelMapping,
                                                                 numTilesPerIpu
    );
    const auto workerGranularityMappings = grids::toWorkerPartitions(
            tileGranularityMappings,
            numWorkers
    );

    grids::serializeToJson(workerGranularityMappings, "partitioning.json");

    auto tensors = std::map < std::string, Tensor>{};
    auto programs = std::vector < Program > {};

    Graph graph(target);

    timedStep("Building computational graph",
              [&]() {
                  popops::addCodelets(graph);

                  graph.addCodelets("codelets/D2Q9Codelets.cpp", CodeletFileType::Auto, debug ? "" : "-O3");

                  tensors["av_vel"] = graph.addVariable(FLOAT, {params->maxIters, 1},
                                                        "av_vel");
                  tensors["cells"] = graph.addVariable(FLOAT, {params->ny, params->nx, lbm::NumSpeeds}, "cells");
                  tensors["tmp_cells"] = graph.addVariable(FLOAT, {params->ny, params->nx, lbm::NumSpeeds},
                                                           "tmp_cells");
                  tensors["obstacles"] = graph.addVariable(BOOL, {params->ny, params->nx}, "obstacles");
                  tensors["perWorkerPartialSums"] = graph.addVariable(FLOAT,
                                                                      {numWorkers * numTilesPerIpu * numIpus},
                                                                      poplar::VariableMappingMethod::LINEAR,
                                                                      "perWorkerPartialSums");
                  tensors["perWorkerPartialCounts"] = graph.addVariable(INT,
                                                                        {numWorkers * numTilesPerIpu * numIpus},
                                                                        poplar::VariableMappingMethod::LINEAR,
                                                                        "perWorkerPartialCounts");

                  tensors["reducedSum"] = graph.addVariable(FLOAT, {}, "reducedSum");
                  graph.setInitialValue(tensors["reducedSum"], 0.f);
                  graph.setTileMapping(tensors["reducedSum"], 0);
                  tensors["reducedCount"] = graph.addVariable(INT, {}, "reducedCount");
                  graph.setTileMapping(tensors["reducedCount"], 0);
                  graph.setInitialValue(tensors["reducedCount"], 0u);

                  mapCellsToTiles(graph, tensors["cells"], tileGranularityMappings);
                  mapCellsToTiles(graph, tensors["tmp_cells"], tileGranularityMappings);
                  mapCellsToTiles(graph, tensors["obstacles"], tileGranularityMappings);

                  tensors["counter"] = graph.addVariable(UNSIGNED_INT, {}, "counter");
                  graph.setTileMapping(tensors["counter"], numTilesPerIpu - 1);
                  graph.setInitialValue(tensors["counter"], 0);


                  auto outStreamAveVelocities = graph.addDeviceToHostFIFO("<<av_vel", FLOAT, params->maxIters);
                  auto outStreamFinalCells = graph.addDeviceToHostFIFO("<<cells", FLOAT,
                                                                       lbm::NumSpeeds * params->nx * params->ny);
                  auto inStreamCells = graph.addHostToDeviceFIFO(">>cells", FLOAT,
                                                                 lbm::NumSpeeds * params->nx * params->ny);
                  auto inStreamObstacles = graph.addHostToDeviceFIFO(">>obstacles", BOOL,
                                                                     params->nx * params->ny);

                  auto copyCellsAndObstaclesToDevice = Sequence(Copy(inStreamCells, tensors["cells"]),
                                                                Copy(inStreamObstacles, tensors["obstacles"]));
                  auto streamBackToHostProg = Sequence(
                          Copy(tensors["cells"], outStreamFinalCells),
                          Copy(tensors["av_vel"], outStreamAveVelocities)
                  );
                  averageVelocity(graph, *params, tensors, workerGranularityMappings);

                  auto prog = Sequence{Repeat(params->maxIters, Sequence{
                          timestep(graph, *params, tensors, workerGranularityMappings),
//                          averageVelocity(graph, *params, tensors, workerGranularityMappings)
                  })};
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


    auto exe = std::optional < Executable > {};
    timedStep("Compiling graph", [&]() -> void {
        ProgressFunc progressFunc = {
                [](int a, int b) -> void { std::cerr << "  Step " << a << " of " << b << std::endl; }};
        exe = {poplar::compileGraph(graph, programs,
                                    debug ? POPLAR_ENGINE_OPTIONS_DEBUG : POPLAR_ENGINE_OPTIONS_NODEBUG,
                                    progressFunc)};
    });

    timedStep("Serializing executable", [&]() -> void {
        ofstream exe_file;
        exe_file.open(outputFilename);
        exe->serialize(exe_file);
        exe_file.close();
    });

    timedStep("Serializing graph for analysis", [&]() -> void {
        serializeGraph(graph);
    });

    std::cout << "==done==" << std::endl;

    return EXIT_SUCCESS;
}
