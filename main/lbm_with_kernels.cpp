

#include <cstdlib>
#include <poplar/IPUModel.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ScaledAdd.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <iomanip>
#include <iostream>
#include <poplar/Program.hpp>
#include <popops/Reduce.hpp>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <poputil/Broadcast.hpp>
#include <random>

#include "DoubleRoll.hpp"
#include "GraphcoreUtils.hpp"
#include "LbmParams.hpp"
#include "LatticeBoltzmann.hpp"

using namespace poplar;
using namespace poplar::program;
using namespace popops;

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

    auto device = lbm::getIpuModel();
    //auto device = getIpuDevice();
    if (!device.has_value()) {
        return EXIT_FAILURE;
    }

    auto tensors = std::map<std::string, Tensor>{};

    double total_compute_time = 0.0;
    std::chrono::high_resolution_clock::time_point tic, toc;

    //------
    std::cerr << "Building computational graph" << std::endl;
    tic = std::chrono::high_resolution_clock::now();

    Graph graph(device.value().getTarget());
    popops::addCodelets(graph);

    tensors["av_vel"] = graph.addVariable(FLOAT, {params->maxIters}, poplar::VariableMappingMethod::LINEAR, "av_vel");
    tensors["cells"] = graph.addVariable(FLOAT, {params->ny, params->nx, lbm::NumSpeeds},
                                         poplar::VariableMappingMethod::LINEAR, "cells");

    toc = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic).count() * 1000;
    std::cerr << "Took " << std::right << std::setw(12) << std::setprecision(5) << diff << "ms" << std::endl;

    //-----

    std::cerr << "Creating engine and loading computational graph" << std::endl;
    tic = std::chrono::high_resolution_clock::now();

    auto outStreamAveVelocities = graph.addDeviceToHostFIFO("<<av_vel", FLOAT, params->maxIters);
    auto outStreamFinalCells = graph.addDeviceToHostFIFO("<<cells", FLOAT, lbm::NumSpeeds * params->nx * params->ny);
    auto inStreamCells = graph.addHostToDeviceFIFO(">>cells", FLOAT, lbm::NumSpeeds * params->nx * params->ny);

    for (const auto &[k, v]: tensors) {
        std::cout << k << std::endl;
    }

    auto copyCellsToDevice = Copy(inStreamCells, tensors["cells"]);
    auto streamBackToHostProg = Sequence(
            Copy(tensors["cells"], outStreamFinalCells),
            Copy(tensors["av_vel"], outStreamAveVelocities)
    );

    auto prog = Sequence();

    auto engine = lbm::createDebugEngine(graph, {copyCellsToDevice, prog, streamBackToHostProg});
    auto av_vels = std::vector<float>(params->maxIters, 0.0f);
    engine.connectStream(outStreamAveVelocities, av_vels.data());
    engine.connectStream(outStreamFinalCells, cells.getData());
    engine.connectStream(inStreamCells, cells.getData());
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
    std::cerr << "took " << std::right << std::setw(12) << std::setprecision(5) << diff << "s" << std::endl;

    std::cerr << "Running copy to host step ";
    tic = std::chrono::high_resolution_clock::now();

    engine.run(2);
    toc = std::chrono::high_resolution_clock::now();
    diff = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic).count();
    std::cerr << "took " << std::right << std::setw(12) << std::setprecision(5) << diff << "s" << std::endl;


    lbm::writeAverageVelocities("av_vels.dat", av_vels);
    lbm::writeResults("final_state.dat", *params, *obstacles, cells);

    lbm::captureProfileInfo(engine);

    engine.printProfileSummary(std::cout,
                               OptionFlags{{"showExecutionSteps", "false"}});

    std::cerr << "Total compute time was  " << std::right << std::setw(12) << std::setprecision(5) << total_compute_time
              << "s" << std::endl;

    return EXIT_SUCCESS;
}
