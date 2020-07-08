

#include <cstdlib>
#include <poplar/IPUModel.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ScaledAdd.hpp>
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


    const auto numIpus = graph.getTarget().getNumIPUs();
    const auto numTilesPerIpu = graph.getTarget().getNumTiles() / numIpus;
    const auto numWorkers = graph.getTarget().getNumWorkerContexts();


    ComputeSet csVel = graph.addComputeSet("normedVelocity");
    ComputeSet csMaskedPartial = graph.addComputeSet("maskedPartial");
    ComputeSet csTileReducedPartial = graph.addComputeSet("perTileReduce");
    ComputeSet csIpuReducedPartial = graph.addComputeSet("perIpuReduce");
    ComputeSet csFinalReduce = graph.addComputeSet("finalReduce");
    ComputeSet csAppendResult = graph.addComputeSet("appendReducedSum");


    for (const auto &[target, slice] : workerLevelMappings) {
        auto tile = target.ipu() * numTilesPerIpu + target.tile();
        auto numCellsForThisWorker = slice.width() * slice.height();
        /* Step 1. On each worker, calculate the the local normed velocity for the slice of cells we own */

        auto normedVelocityVertex = graph.addVertex(csVel,
                                                    "NormedVelocityVertex",
                                                    {{"cells",    applySlice(tensors["cells"], slice)},
                                                     {"numCells", numCellsForThisWorker},
                                                     {"vels",     applySlice(tensors["velocities"], slice)}});
        graph.setCycleEstimate(normedVelocityVertex, numCellsForThisWorker);
        graph.setTileMapping(normedVelocityVertex, tile);

        auto toPartialIdx = [=](grids::MappingTarget target) -> size_t {
            auto seq_worker = target.ipu() * numTilesPerIpu * numWorkers + target.tile() * numWorkers + target.worker();
            return seq_worker;
        };
        auto partialSlice = grids::Slice2D{{toPartialIdx(target), toPartialIdx(target) + 1},
                                           {0,                    2}};
        /* Step 2. On each worker, calculate the the masked sum and count (i.e. only those where no obstacle) */
        auto maskedSumPartialVertex = graph.addVertex(csMaskedPartial,
                                                      "MaskedSumPartial",
                                                      {{"velocities",    applySlice(tensors["velocities"], slice)},
                                                       {"numCells",      numCellsForThisWorker},
                                                       {"totalAndCount", applySlice(tensors["perWorkerPartials"],
                                                                                    partialSlice)},
                                                       {"obstacles",     applySlice(tensors["obstacles"], slice)}}
        );
        graph.setCycleEstimate(maskedSumPartialVertex, numCellsForThisWorker);
        graph.setTileMapping(maskedSumPartialVertex, tile);
    }

    /* Step 3. On each tile, calculate sum and count of all workers */
    for (auto ipu = 0u; ipu < numIpus; ipu++) {
        for (auto tile = 0u; tile < numTilesPerIpu; tile++) {
            auto seq_tile = ipu * numTilesPerIpu + tile;
            auto allWorkerPartialsSlice = grids::Slice2D{{seq_tile * numWorkers, (seq_tile + 1) * numWorkers},
                                                         {0,                     2}};
            auto resultSlice = grids::Slice2D{{seq_tile, seq_tile + 1},
                                              {0,        2}};
            auto perTileReduceVertex = graph.addVertex(csTileReducedPartial,
                                                       "ReducePartials",
                                                       {{"totalAndCountPartials", applySlice(
                                                               tensors["perWorkerPartials"],
                                                               allWorkerPartialsSlice)},
                                                        {"numPartials",           numWorkers},
                                                        {"totalAndCount",         applySlice(tensors["perTilePartials"],
                                                                                             resultSlice)},
                                                       }
            );
            graph.setCycleEstimate(perTileReduceVertex, numWorkers * 4);
            graph.setTileMapping(perTileReduceVertex, tile);
        }
    }

    /* Step 4. On each IPU, calculate sum and count of all tiles */
    for (auto ipu = 0u; ipu < numIpus; ipu++) {
        auto allTilesPartialsSlice = grids::Slice2D{{ipu * numTilesPerIpu, (ipu + 1) * numTilesPerIpu},
                                                    {0,                    2}};
        auto perIpuReduceVertex = graph.addVertex(csIpuReducedPartial,
                                                  "ReducePartials",
                                                  {{"totalAndCountPartials", applySlice(tensors["perTilePartials"],
                                                                                        allTilesPartialsSlice)},
                                                   {"numPartials",           numTilesPerIpu},
                                                   {"totalAndCount",         tensors["perIpuPartials"].slice(ipu,
                                                                                                             ipu + 1,
                                                                                                             0).flatten()},
                                                  }
        );
        graph.setCycleEstimate(perIpuReduceVertex, numTilesPerIpu);
        graph.setTileMapping(perIpuReduceVertex, ipu * numTilesPerIpu);
    }

    /* Step 5. Calculate the total and count over all IPUs */
    auto finalReduceVertex = graph.addVertex(csFinalReduce,
                                             "ReducePartials",
                                             {{"totalAndCountPartials", tensors["perIpuPartials"].flatten()},
                                              {"numPartials",           numIpus},
                                              {"totalAndCount",         tensors["reducedSumAndCount"].flatten()},
                                             }
    );
    graph.setCycleEstimate(finalReduceVertex, numWorkers);
    graph.setTileMapping(finalReduceVertex, 0);

    /* Step 6. Calculate the average and write it to the relevant place in the array. This happens on every tile,
     * because each tile owns a piece of cells, and only the owner of the piece with the index actually writes */
    auto avVelsTileMapping = grids::partitionGridToTilesForSingleIpu(
            {params.maxIters, 1},
            numTilesPerIpu * numIpus
    );

    for (const auto &[target, slice] : avVelsTileMapping) {
        auto tile = target.ipu() * numTilesPerIpu + target.tile();

        auto appendReducedSumVertex = graph.addVertex(csAppendResult,
                                                      "AppendReducedSum",
                                                      {{"totalAndCount", tensors["reducedSumAndCount"].flatten()},
                                                       {"indexToWrite",  tensors["counter"]},
                                                       {"myStartIndex",  slice.rows().from()},
                                                       {"myEndIndex",    slice.rows().to() - 1},
                                                       {"finals",        tensors["av_vel"].slice(slice.rows().from(),
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


    return Sequence(Execute(csVel), Execute(csMaskedPartial), Execute(csTileReducedPartial),
                    Execute(csIpuReducedPartial), Execute(csFinalReduce), Execute(csAppendResult),
                    Execute(incrementCs));
}

auto
collision(Graph &graph, const lbm::Params &params, TensorMap &tensors, grids::TileMappings &mappings) -> Program {
    const auto numTilesPerIpu = graph.getTarget().getNumTiles() / graph.getTarget().getNumIPUs();

    ComputeSet reboundCs = graph.addComputeSet("collision");

    for (const auto &[target, slice] : mappings) {
        auto tile = target.ipu() * numTilesPerIpu + target.tile();
        auto numCellsForThisWorker = slice.width() * slice.height();

        auto v = graph.addVertex(reboundCs,
                                 "CollisionVertex",
                                 {
                                         {"in",        applySlice(tensors["tmp_cells"], slice)},
                                         {"out",       applySlice(tensors["cells"], slice)},
                                         {"numRows",   slice.height()},
                                         {"numCols",   slice.width()},
                                         {"omega",     params.omega},
                                         {"obstacles", applySlice(tensors["obstacles"], slice)},
                                 });
        graph.setCycleEstimate(v, numCellsForThisWorker);
        graph.setTileMapping(v, tile);
    }
    return Execute(reboundCs);
}

auto rebound(Graph &graph, const lbm::Params &params, TensorMap &tensors, grids::TileMappings &mappings) -> Program {
    const auto numTilesPerIpu = graph.getTarget().getNumTiles() / graph.getTarget().getNumIPUs();

    ComputeSet reboundCs = graph.addComputeSet("rebound");

    for (const auto &[target, slice] : mappings) {
        auto tile = target.ipu() * numTilesPerIpu + target.tile();
        auto numCellsForThisWorker = slice.width() * slice.height();
        auto v = graph.addVertex(reboundCs,
                                 "ReboundVertex",
                                 {
                                         {"in",        applySlice(tensors["tmp_cells"], slice)},
                                         {"out",       applySlice(tensors["cells"], slice)},
                                         {"numRows",   slice.height()},
                                         {"numCols",   slice.width()},
                                         {"obstacles", applySlice(tensors["obstacles"], slice)},
                                 });
        graph.setCycleEstimate(v, numCellsForThisWorker);
        graph.setTileMapping(v, tile);
    }
    return Execute(reboundCs);
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


auto timestep(Graph &graph, const lbm::Params &params, TensorMap &tensors, grids::TileMappings &mappings) -> Program {
    return Sequence{
            accelerate_flow(graph, params, tensors),
            propagate(graph, params, tensors, mappings),
            rebound(graph, params, tensors, mappings),
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


    // auto device = lbm::getIpuModel();
    auto device = lbm::getIpuDevice();
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

    double total_compute_time = 0.0;
    std::chrono::high_resolution_clock::time_point tic, toc;

    //------
    std::cerr << "Building computational graph" << std::endl;
    tic = std::chrono::high_resolution_clock::now();

    Graph graph(device.value().getTarget());
    popops::addCodelets(graph);
    graph.addCodelets("D2Q9Codelets.cpp");

    tensors["av_vel"] = graph.addVariable(FLOAT, {params->maxIters, 1},
                                          "av_vel");
    tensors["cells"] = graph.addVariable(FLOAT, {params->ny, params->nx, lbm::NumSpeeds}, "cells");
    tensors["tmp_cells"] = graph.addVariable(FLOAT, {params->ny, params->nx, lbm::NumSpeeds}, "tmp_cells");
    tensors["obstacles"] = graph.addVariable(BOOL, {params->ny, params->nx}, "obstacles");
    tensors["velocities"] = graph.addVariable(FLOAT, {params->ny, params->nx}, "velocities");
    tensors["perWorkerPartials"] = graph.addVariable(FLOAT,
                                                     {numWorkers * numTilesPerIpu * numIpus, 2},
                                                     poplar::VariableMappingMethod::LINEAR,
                                                     "perWorkerPartials");

    tensors["perTilePartials"] = graph.addVariable(FLOAT,
                                                   {numTilesPerIpu * numIpus, 2}, poplar::VariableMappingMethod::LINEAR,
                                                   "perTilePartials");

    tensors["perIpuPartials"] = graph.addVariable(FLOAT,
                                                  {numIpus, 2},
                                                  "perTilePartials");
    for (auto i = 0u; i < numIpus; i++) {
        graph.setTileMapping(tensors["perIpuPartials"][i], i * numTilesPerIpu);
    }

    tensors["reducedSumAndCount"] = graph.addVariable(FLOAT, {1, 2}, "reducedSumAndCount");
    graph.setTileMapping(tensors["reducedSumAndCount"], 0);

    mapCellsToTiles(graph, tensors["cells"], tileGranularityMappings);
    mapCellsToTiles(graph, tensors["tmp_cells"], tileGranularityMappings);
    mapCellsToTiles(graph, tensors["obstacles"], tileGranularityMappings);
    mapCellsToTiles(graph, tensors["velocities"], tileGranularityMappings);

    tensors["counter"] = graph.addVariable(UNSIGNED_INT, {}, "counter");
    graph.setTileMapping(tensors["counter"], 0);
    graph.setInitialValue(tensors["counter"], 0);

    toc = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic).count() * 1000;
    std::cerr << "Took " << std::right << std::setw(12) << std::setprecision(5) << diff << "ms" << std::endl;

    //-----



    std::cerr << "Creating engine and loading computational graph" << std::endl;
    tic = std::chrono::high_resolution_clock::now();

    auto outStreamAveVelocities = graph.addDeviceToHostFIFO("<<av_vel", FLOAT, params->maxIters);
    auto outStreamFinalCells = graph.addDeviceToHostFIFO("<<cells", FLOAT,
                                                         lbm::NumSpeeds * params->nx * params->ny);
    auto inStreamCells = graph.addHostToDeviceFIFO(">>cells", FLOAT, lbm::NumSpeeds * params->nx * params->ny);
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

    auto engine = lbm::createDebugEngine(graph, {copyCellsAndObstaclesToDevice, prog, streamBackToHostProg});
    auto av_vels = std::vector<float>(params->maxIters, 0.0f);
    engine.connectStream(outStreamAveVelocities, av_vels.data());
    engine.connectStream(outStreamFinalCells, cells.getData());
    engine.connectStream(inStreamCells, cells.getData());
    engine.connectStream(inStreamObstacles, obstacles->getData());
    std::cerr << "Loading..." << std::endl;


    engine.load(device.value());

    toc = std::chrono::high_resolution_clock::now();

    diff = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic).count();
    std::cerr << "Took " << std::right << std::setw(12) << std::setprecision(5) << diff << "s" << std::endl;

    std::cerr << "Running copy to device step ";
    tic = std::chrono::high_resolution_clock::now();

    engine.run(0);
    toc = std::chrono::high_resolution_clock::now();
    diff = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic).count();
    std::cerr << "took " << std::right << std::setw(12) << std::setprecision(5) << diff << "s" << std::endl;

    std::cerr << "Running LBM ";
    tic = std::chrono::high_resolution_clock::now();

    engine.run(1);
    toc = std::chrono::high_resolution_clock::now();
    diff = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic).count();
    total_compute_time += diff;
    std::cerr << "took " << std::right << std::setw(12) << std::setprecision(5) << diff << "s" << std::endl;

    std::cerr << "Running copy to host step ";
    tic = std::chrono::high_resolution_clock::now();

    engine.run(2);
    toc = std::chrono::high_resolution_clock::now();
    diff = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic).count();
    std::cerr << "took " << std::right << std::setw(12) << std::setprecision(5) << diff << "s" << std::endl;


    lbm::writeAverageVelocities("av_vels.dat", av_vels);
    lbm::writeResults("final_state.dat", *params, *obstacles, cells);

   lbm::captureProfileInfo(engine, graph);
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
