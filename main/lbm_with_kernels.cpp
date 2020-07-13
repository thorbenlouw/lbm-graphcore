

#include <cstdlib>
#include <poplar/IPUModel.hpp>
#include <popops/Reduce.hpp>
#include <popops/codelets.hpp>
#include <iomanip>
#include <iostream>
#include <poplar/Program.hpp>
#include <chrono>
#include <algorithm>
#include <poputil/Broadcast.hpp>
#include <random>

#include "GraphcoreUtils.hpp"
#include "LbmParams.hpp"
#include "LatticeBoltzmann.hpp"
#include "StructuredGridUtils.hpp"


using namespace poplar;
using namespace poplar::program;
using namespace popops;

using TensorMap = std::map<std::string, Tensor>;

auto applySlice(Tensor &tensor, grids::Slice2D slice) -> Tensor {
    return
            tensor.slice(slice.rows().from(), slice.rows().to(), 0)
                    .slice(slice.cols().from(), slice.cols().to(),
                           1).flatten();
};

auto
averageVelocity(Graph &graph, const lbm::Params &params, TensorMap &tensors,
                const grids::GridPartitioning &workerLevelMappings) -> Program {

    // As part of collision we already calculated a partialSum (float) and partialCount (unsigned) for each worker
    // which represents the summed normedVelocity and count of cells which are not masked by obstacles. Now we reduce them

    // Do multiple reductions in parallel
    std::vector<ComputeSet> reductionComputeSets;
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

    auto ipuPartitioning = grids::partitionForIpus({params.maxIters, 1}, numIpus, params.maxIters);
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
    graph.setTileMapping(incrementVertex, 0);


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
                                         {"in",                    applySlice(tensors["tmp_cells"], slice)},
                                         {"out",                   applySlice(tensors["cells"], slice)},
                                         {"numRows",               slice.height()},
                                         {"numCols",               slice.width()},
                                         {"omega",                 params.omega},
                                         {"obstacles",             applySlice(tensors["obstacles"], slice)},
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
        auto halos = grids::Halos::forSlice(slice, fullSize);
        auto v = graph.addVertex(propagateCs,
                                 "PropagateVertex",
                                 {
                                         {"in",              applySlice(cells, slice)},
                                         {"out",             applySlice(tensors["tmp_cells"], slice)},
                                         {"numRows",         slice.height()},
                                         {"numCols",         slice.width()},
                                         {"haloTop",         applySlice(cells, *halos.top)},
                                         {"haloBottom",      applySlice(cells, *halos.bottom)},
                                         {"haloLeft",        applySlice(cells, *halos.left)},
                                         {"haloRight",       applySlice(cells, *halos.right)},
                                         {"haloTopLeft",     applySlice(cells,
                                                                        *halos.topLeft)[lbm::SpeedIndexes::SouthEast]}, // flipped directions!
                                         {"haloTopRight",    applySlice(cells,
                                                                        *halos.topRight)[lbm::SpeedIndexes::SouthWest]},// flipped directions!
                                         {"haloBottomLeft",  applySlice(cells,
                                                                        *halos.bottomLeft)[lbm::SpeedIndexes::NorthEast]},// flipped directions!
                                         {"haloBottomRight", applySlice(cells,
                                                                        *halos.bottomRight)[lbm::SpeedIndexes::NorthWest]},// flipped directions!
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

    // For now, let's try the approach of spreading accelerate computation over more tiles, even if that
    // means redistributing data from cells and obstacles (i.e. keep tiles busy rather than minimise data transfer)
    // TODO actually just run on the tiles where this stuff is mapped
    const auto ipuLevelMapping = grids::partitionForIpus({1, params.nx}, graph.getTarget().getNumIPUs(), params.nx);

    assert(ipuLevelMapping.has_value());
    auto tileGranularityMappings = grids::toTilePartitions(
            *ipuLevelMapping,
            numTilesPerIpu
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
                                 {{"cellsInSecondRow",     applySlice(cellsSecondRowFromTop, slice)},
                                  {"obstaclesInSecondRow", applySlice(obstaclesSecondRowFromTop, slice)},
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
            accelerate_flow(graph, params, tensors),
            propagate(graph, params, tensors, mappings),
            collision(graph, params, tensors, mappings)
    };
}

auto compile(int argc, char *argv[]) -> int {
    if (argc < 6) {
        std::cerr << "Expected usage: " << argv[0]
                  << " compile <device:{model|ipu}> <num_ipus> <params_file> <output_file>"
                  << std::endl;
        return EXIT_FAILURE;
    }
    const auto deviceArg = std::string{argv[2]};
    const auto numIpusArg = std::stoi(std::string{argv[3]});
    const auto paramsFileArg = std::string{argv[4]};
    const auto outputFilename = std::string{argv[5]};

    auto params = lbm::Params::fromFile(paramsFileArg);
    if (!params.has_value()) {
        std::cerr << "Could not parse parameters file. Aborting" << std::endl;
        return EXIT_FAILURE;
    }

    auto device = deviceArg == "ipu" ? utils::getIpuDevice(numIpusArg) : utils::getIpuModel(numIpusArg);
    if (!device.has_value()) {
        return EXIT_FAILURE;
    }

    auto numTilesPerIpu = device->getTarget().getNumTiles() / device->getTarget().getNumIPUs();
    auto numWorkers = device->getTarget().getNumWorkerContexts();
    auto numIpus = device->getTarget().getNumIPUs();

    const auto ipuLevelMapping = grids::partitionForIpus({params->ny, params->nx}, numIpus, 2000 * 1000);
    if (!ipuLevelMapping.has_value()) {
        std::cerr << "Couldn't find a way to partition the input parameter space over the given number of IPUs"
                  << std::endl;
        return EXIT_FAILURE;
    }
    const auto tileGranularityMappings = grids::toTilePartitions(*ipuLevelMapping,
                                                                 numTilesPerIpu,
                                                                 6,
                                                                 6
    );
    const auto workerGranularityMappings = grids::toWorkerPartitions(
            tileGranularityMappings,
            numWorkers
    );

    grids::serializeToJson(workerGranularityMappings, "/tmp/partitioning.json");

    auto tensors = std::map<std::string, Tensor>{};
    auto programs = std::vector<Program>{};

    Graph graph(device.value().getTarget());

    utils::timedStep("Building computational graph",
                     [&]() {
                         popops::addCodelets(graph);

                         graph.addCodelets("D2Q9Codelets.cpp");

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

                         utils::mapCellsToTiles(graph, tensors["cells"], tileGranularityMappings);
                         utils::mapCellsToTiles(graph, tensors["tmp_cells"], tileGranularityMappings);
                         utils::mapCellsToTiles(graph, tensors["obstacles"], tileGranularityMappings);

                         tensors["counter"] = graph.addVariable(UNSIGNED_INT, {}, "counter");
                         graph.setTileMapping(tensors["counter"], 0);
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

                         auto prog = Repeat(params->maxIters, Sequence{
                                 timestep(graph, *params, tensors, workerGranularityMappings),
                                 averageVelocity(graph, *params, tensors, workerGranularityMappings)
                         });
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


    auto exe = std::optional<Executable>{};
    utils::timedStep("Compiling graph", [&]() -> void {
        ProgressFunc progressFunc = {[](int a, int b) -> void { std::cerr << a << "/" << b << std::endl; }};
        exe = {poplar::compileGraph(graph, programs,
                                    utils::POPLAR_ENGINE_OPTIONS_DEBUG,
                                    progressFunc)};
    });

    utils::timedStep("Serializing executable", [&]() -> void {
        ofstream exe_file;
        exe_file.open(outputFilename);
        exe->serialize(exe_file);
        exe_file.close();
    });

    utils::timedStep("Serializing executable", [&]() -> void {
        utils::serializeGraph(graph);
    });

    std::cout << "==done==" << std::endl;

    return EXIT_SUCCESS;
}

auto run(int argc, char *argv[]) -> int {
    if (argc < 6) {
        std::cerr << "Expected usage: " << argv[0]
                  << " run  <device:{model|ipu}> <num_ipus> <compiled_graph> <params_file> <obstacles_file>"
                  << std::endl;
        return EXIT_FAILURE;
    }
    const auto graphFilenameArg = std::string{argv[4]};
    const auto paramsFileArg = std::string{argv[5]};
    const auto obstaclesFileArg = std::string{argv[6]};

    const auto deviceArg = std::string{argv[2]};
    const auto numIpusArg = std::stoi(std::string{argv[3]});


    auto params = lbm::Params::fromFile(paramsFileArg);
    if (!params.has_value()) {
        std::cerr << "Could not parse parameters file. Aborting" << std::endl;
        return EXIT_FAILURE;
    }

    auto obstacles = lbm::Obstacles::fromFile(params->nx, params->ny, argv[2]);
    if (!obstacles.has_value()) {
        std::cerr << "Could not parse obstacles file" << std::endl;
        return EXIT_FAILURE;
    }

    auto device = deviceArg == "ipu" ? utils::getIpuDevice(numIpusArg) : utils::getIpuModel(numIpusArg);
    if (!device.has_value()) {
        return EXIT_FAILURE;
    }

    double total_compute_time = 0.0;


    auto cells = lbm::Cells(params->nx, params->ny);
    cells.initialise(*params);


    std::unique_ptr<Engine> engine;
    auto av_vels = std::vector<float>(params->maxIters, 0.0f);


    ifstream exe_file;
    exe_file.open(graphFilenameArg);
    auto exe = poplar::Executable::deserialize(exe_file);
    exe_file.close();

    utils::timedStep("Creating engine and connecting streams", [&]() {
        engine = std::make_unique<Engine>(exe, utils::POPLAR_ENGINE_OPTIONS_DEBUG);

        engine->connectStream("<<av_vel", av_vels.data());
        engine->connectStream("<<cells", cells.getData());
        engine->connectStream(">>cells", cells.getData());
        engine->connectStream(">>obstacles", obstacles->getData());

    });

    utils::timedStep("Loading graph to device", [&]() {
        engine->load(device.value());
    });

    utils::timedStep("Running copy to device step", [&]() {
        engine->run(0);
    });

    total_compute_time += utils::timedStep("Running LBM", [&]() {
        engine->run(1);
    });

    utils::timedStep("Running copy to host step", [&]() {
        engine->run(2);
    });

    utils::timedStep("Writing output files ", [&]() {
        lbm::writeAverageVelocities("av_vels.dat", av_vels);
        lbm::writeResults("final_state.dat", *params, *obstacles, cells);
    });

    utils::timedStep("Capturing profiling info", [&]() {
        utils::captureProfileInfo(*engine);
    });

//
//    engine.printProfileSummary(std::cout,
//                               OptionFlags{{"showExecutionSteps", "true"}});


    std::cout << "==done==" << std::endl;
    std::cout << "Total compute time was \t" << std::right << std::setw(12) << std::setprecision(5)
              << total_compute_time
              << "s" << std::endl;

    std::cout << "Reynolds number:  \t" << std::right << std::setw(12) << std::setprecision(12) << std::scientific
              << lbm::reynoldsNumber(*params, av_vels[params->maxIters - 1]) << std::endl;

    return EXIT_SUCCESS;
}

auto main(int argc, char *argv[]) -> int {
    assert(argc > 1);
    auto command = argv[1];
    if (strcmp(command, "compile") == 0) {
        return compile(argc, argv);
    } else if (strcmp(command, "run") == 0) {
        return run(argc, argv);
    }
    return EXIT_FAILURE;

}
