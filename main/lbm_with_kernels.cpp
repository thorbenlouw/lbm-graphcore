

#include <cstdlib>
#include <poplar/IPUModel.hpp>
#include <popops/ElementWise.hpp>
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
                grids::TileMappings &workerLevelMappings) -> Program {

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

    auto avVelsTileMapping = grids::partitionGridToTilesForSingleIpu(
            {params.maxIters, 1},
            numTilesPerIpu * numIpus
    );

    ComputeSet appendResultCs = graph.addComputeSet("appendReducedSum");
    for (const auto &[target, slice] : avVelsTileMapping) {
        auto tile = target.ipu() * numTilesPerIpu + target.tile();

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
collision(Graph &graph, const lbm::Params &params, TensorMap &tensors, grids::TileMappings &mappings) -> Program {
    const auto numTilesPerIpu = graph.getTarget().getNumTiles() / graph.getTarget().getNumIPUs();
    const auto numWorkersPerTile = graph.getTarget().getNumWorkerContexts();

    ComputeSet collisionCs = graph.addComputeSet("collision");

    for (const auto &[target, slice] : mappings) {
        auto tile = target.ipu() * numTilesPerIpu + target.tile();
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
               grids::TileMappings &mappings) -> Program {
    const auto numTilesPerIpu = graph.getTarget().getNumTiles() / graph.getTarget().getNumIPUs();

    ComputeSet propagateCs = graph.addComputeSet("propagate");
    auto cells = tensors["cells"];

    auto fullSize = grids::Size2D(params.ny, params.nx);
    for (const auto &[target, slice] : mappings) {
        auto tile = target.ipu() * numTilesPerIpu + target.tile();
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
    auto tileGranularityMappings = grids::partitionGridToTilesForSingleIpu(
            {1, params.nx},
            numTilesPerIpu
    );
    auto workerGranularityMappings = grids::toWorkerMappings(
            tileGranularityMappings,
            numWorkers
    );

    for (const auto &[target, slice] : workerGranularityMappings) {

        auto tile = target.ipu() * numTilesPerIpu + target.tile();

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
timestep(Graph &graph, const lbm::Params &params, TensorMap &tensors, grids::TileMappings &mappings) -> Program {
    return Sequence{
            accelerate_flow(graph, params, tensors),
            propagate(graph, params, tensors, mappings),
            collision(graph, params, tensors, mappings)
    };
}

auto mapCellsToTiles(Graph &graph, Tensor &cells, grids::TileMappings &tileMappings, bool print = false) {
    const auto numTilesPerIpu = graph.getTarget().getNumTiles();
    for (const auto&[target, slice]: tileMappings) {
        const auto tile = target.ipu() * numTilesPerIpu + target.tile();

        if (print) {
            std::cout << "tile: " << tile << " target: " << target.ipu() << ":" << target.tile() << ":"
                      << target.worker() <<
                      "(r: " << slice.rows().from() << ",c: " << slice.cols().from() << ",w: " << slice.width() <<
                      ",h: " << slice.height() << std::endl;
        }
        graph.setTileMapping(cells
                                     .slice(slice.rows().from(), slice.rows().to(), 0)
                                     .slice(slice.cols().from(), slice.cols().to(), 1),
                             tile);
    }
}


auto main(int argc, char *argv[]) -> int {
    double total_compute_time = 0.0;

    auto timedStep = [&total_compute_time](const std::string description, auto f,
                                           bool addToComputeTime = false) {
        std::cerr << std::setw(60) << description;
        auto tic = std::chrono::high_resolution_clock::now();
        f();
        auto toc = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic).count();
        std::cerr << " took " << std::right << std::setw(12) << std::setprecision(5) << diff << "s" << std::endl;
        if (addToComputeTime) total_compute_time += diff;
    };

    if (argc != 3) {
        std::cerr << "Expected usage: " << argv[0] << " <params_file> <obstacles_file>" << std::endl;
        return EXIT_FAILURE;
    }
    auto params = lbm::Params::fromFile(argv[1]);
    if (!params.has_value()) {
        std::cerr << "Could not parse parameters file. Aborting" << std::endl;
        return EXIT_FAILURE;
    }
    auto obstacles = lbm::Obstacles::fromFile(params->nx, params->ny, argv[2]);
    if (!obstacles.has_value()) {
        std::cerr << "Could not parse obstacles file" << std::endl;
        return EXIT_FAILURE;
    }

    auto cells = lbm::Cells(params->nx, params->ny);
    cells.initialise(*params);


    auto device = lbm::getIpuModel();
//    auto device = lbm::getIpuDevice();
    if (!device.has_value()) {
        return EXIT_FAILURE;
    }


    auto numTilesPerIpu = device->getTarget().getNumTiles() / device->getTarget().getNumIPUs();
    auto numWorkers = device->getTarget().getNumWorkerContexts();
    auto numIpus = device->getTarget().getNumIPUs();

    auto tileGranularityMappings = grids::partitionGridToTilesForSingleIpu(
            {params->ny, params->nx},
            numTilesPerIpu
    );
    auto workerGranularityMappings = grids::toWorkerMappings(
            tileGranularityMappings,
            numWorkers
    );

    auto tensors = std::map<std::string, Tensor>{};

    std::chrono::high_resolution_clock::time_point tic, toc;

    //------
    Graph graph(device.value().getTarget());

    timedStep("Building computational graph",
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


                  mapCellsToTiles(graph, tensors["cells"], tileGranularityMappings);
                  mapCellsToTiles(graph, tensors["tmp_cells"], tileGranularityMappings);
                  mapCellsToTiles(graph, tensors["obstacles"], tileGranularityMappings);

                  tensors["counter"] = graph.addVariable(UNSIGNED_INT, {}, "counter");
                  graph.setTileMapping(tensors["counter"], 0);
                  graph.setInitialValue(tensors["counter"], 0);
              });

    std::unique_ptr<Engine> engine;
    auto av_vels = std::vector<float>(params->maxIters, 0.0f);

    timedStep("Creating engine and loading computational graph", [&]() {

        auto outStreamAveVelocities = graph.addDeviceToHostFIFO("<<av_vel", FLOAT, params->maxIters);
        auto outStreamFinalCells = graph.addDeviceToHostFIFO("<<cells", FLOAT,
                                                             lbm::NumSpeeds * params->nx * params->ny);
        auto inStreamCells = graph.addHostToDeviceFIFO(">>cells", FLOAT,
                                                       lbm::NumSpeeds * params->nx * params->ny);
        auto inStreamObstacles = graph.addHostToDeviceFIFO(">>obstacles", BOOL, params->nx * params->ny);

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

        engine = std::unique_ptr<Engine>(
                new Engine(graph, {copyCellsAndObstaclesToDevice, prog, streamBackToHostProg},
                           lbm::POPLAR_ENGINE_OPTIONS_DEBUG));
        engine->connectStream(outStreamAveVelocities, av_vels.data());
        engine->connectStream(outStreamFinalCells, cells.getData());
        engine->connectStream(inStreamCells, cells.getData());
        engine->connectStream(inStreamObstacles, obstacles->getData());

        engine->load(device.value());
    });

    timedStep("Running copy to device step", [&]() {
        engine->run(0);
    });

    timedStep("Running LBM", [&]() {
        engine->run(1);
    }, true);

    timedStep("Running copy to host step", [&]() {
        engine->run(2);
    });

    timedStep("Writing output files ", [&]() {
        lbm::writeAverageVelocities("av_vels.dat", av_vels);
        lbm::writeResults("final_state.dat", *params, *obstacles, cells);
    });


    timedStep("Capturing profiling info", [&]() {
        lbm::captureProfileInfo(*engine, graph);
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
