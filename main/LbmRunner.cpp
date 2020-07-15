#include "GraphcoreUtils.hpp"
#include "LbmParams.hpp"
#include "LatticeBoltzmann.hpp"
#include "StructuredGridUtils.hpp"
#include <cxxopts.hpp>
#include <poplar/Engine.hpp>

using namespace poplar;
using namespace poplar::program;

auto main(int argc, char *argv[]) -> int {
    std::string graphFilenameArg, paramsFileArg, obstaclesFileArg, deviceArg;

    cxxopts::Options options(argv[0], " - Runs a Lattice Boltzmann graph on the IPU or IPU Model");
    options.add_options()
            ("p,profile", "Capture profiling")
            ("device", "ipu or ipumodel", cxxopts::value<std::string>(deviceArg)->default_value("ipu"))
            ("n,num-ipus", "number of ipus to use", cxxopts::value<unsigned>()->default_value("1"))
            ("exe", "filename of compiled graph", cxxopts::value<std::string>(graphFilenameArg))
            ("params", "filename of parameters file", cxxopts::value<std::string>(paramsFileArg))
            ("obstacles", "filename of obstacles file", cxxopts::value<std::string>(obstaclesFileArg));
    auto opts = options.parse(argc, argv);

    auto captureProfile = opts["profile"].as<bool>();
    if (captureProfile) {
        std::cout << "Capturing profile information during this run." << std::endl;
    };
    auto numIpusArg = opts["num-ipus"].as<unsigned>();

    auto params = lbm::Params::fromFile(paramsFileArg);
    if (!params.has_value()) {
        std::cerr << "Could not parse parameters file. Aborting" << std::endl;
        return EXIT_FAILURE;
    }

    auto obstacles = lbm::Obstacles::fromFile(params->nx, params->ny, obstaclesFileArg);
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

    auto av_vels = std::vector<float>(params->maxIters, 0.0f);


    std::cerr << std::setw(60) << "Deserialising computation graph and creating Engine";
    auto tic = std::chrono::high_resolution_clock::now();
    ifstream exe_file;
    exe_file.open(graphFilenameArg);

    auto engine = Engine(poplar::Executable::deserialize(exe_file),
                         captureProfile ? OptionFlags{{"autoReport.all",   "true"},
                                                      {"debug.instrument", "true"}}
                                        : utils::POPLAR_ENGINE_OPTIONS_NODEBUG);
    exe_file.close();
    engine.connectStream("<<av_vel", av_vels.data());
    engine.connectStream("<<cells", cells.getData());
    engine.connectStream(">>cells", cells.getData());
    engine.connectStream(">>obstacles", obstacles->getData());
    auto toc = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic).count();
    std::cerr << " took " << std::right << std::setw(12) << std::setprecision(5) << diff << "s" << std::endl;


    utils::timedStep("Loading graph to device", [&]() {
        engine.load(device.value());
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

    utils::timedStep("Writing output files ", [&]() {
        lbm::writeAverageVelocities("av_vels.dat", av_vels);
        lbm::writeResults("final_state.dat", *params, *obstacles, cells);
    });

    if (captureProfile) {
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

    std::cout << "Reynolds number:  \t" << std::right << std::setw(12) << std::setprecision(12) << std::scientific
              << lbm::reynoldsNumber(*params, av_vels[params->maxIters - 1]) << std::endl;

    return EXIT_SUCCESS;
}
