#include "include/GraphcoreUtils.hpp"
#include "include/LbmParams.hpp"
#include "include/LatticeBoltzmannUtils.hpp"
#include "include/StructuredGridUtils.hpp"
#include <cxxopts.hpp>
#include <poplar/Engine.hpp>

using namespace poplar;
using namespace poplar::program;

auto main(int argc, char *argv[]) -> int {
    std::string graphFilenameArg, paramsFileArg, obstaclesFileArg, deviceArg;
    unsigned numIpusArg;
    bool debug = false;
    cxxopts::Options options(argv[0], " - Runs a Lattice Boltzmann graph on the IPU or IPU Model");
    options.add_options()
            ("d,debug", "Capture profiling")
            ("device", "ipu or ipumodel", cxxopts::value<std::string>(deviceArg)->default_value("ipu"))
            ("n,num-ipus", "number of ipus to use", cxxopts::value<unsigned>(numIpusArg)->default_value("1"))
            ("exe", "filename of compiled graph", cxxopts::value<std::string>(graphFilenameArg))
            ("params", "filename of parameters file", cxxopts::value<std::string>(paramsFileArg))
            ("obstacles", "filename of obstacles file", cxxopts::value<std::string>(obstaclesFileArg));
    try {
        auto opts = options.parse(argc, argv);
        debug = opts["debug"].as<bool>();
        if (opts.count("n") + opts.count("params") + opts.count("num-ipus") + opts.count("obstacles") + opts.count("exe") < 5) {
            std::cerr << options.help() << std::endl;
            return EXIT_FAILURE;
        }
    } catch (cxxopts::OptionParseException &) {
        std::cerr << options.help() << std::endl;
        return EXIT_FAILURE;
    }
    if (debug) {
        std::cout << "Capturing profile information during this run." << std::endl;
    };

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

    if (debug) {
        std::cout << "Writing debug and profiling info" << std::endl;

    }
    if (deviceArg == "ipu") {
        std::cout << "Running on actual IPU" << std::endl;
    } else {
        std::cout << "Running on IPU emulator" << std::endl;
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

//    auto engine = Engine(poplar::Executable::deserialize(exe_file),
//                         debug ? utils::POPLAR_ENGINE_OPTIONS_DEBUG
//                               : utils::POPLAR_ENGINE_OPTIONS_NODEBUG);
    auto engine = Engine(poplar::Executable::deserialize(exe_file),
                         utils::POPLAR_ENGINE_OPTIONS_DEBUG
                               );
    exe_file.close();
    engine.connectStream("<<av_vel", av_vels.data());
    engine.connectStream("<<cells", cells.data.data());
    engine.connectStream(">>cells", cells.data.data());
    engine.connectStream(">>obstacles", obstacles->getData());
    auto toc = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast < std::chrono::duration < double >> (toc - tic).count();
    std::cerr << " took " << std::right << std::setw(12) << std::setprecision(5) << diff << "s" << std::endl;


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

    utils::timedStep("Writing output files ", [&]() {
        lbm::writeAverageVelocities("av_vels.dat", av_vels);
        lbm::writeResults("final_state.dat", *params, *obstacles, cells);
    });

    if (debug) {
//        utils::timedStep("Capturing profiling info", [&]() {
//            utils::captureProfileInfo(engine);
//        });

        engine.printProfileSummary(std::cout,
                                   OptionFlags{{"showExecutionSteps", "false"}});
    }

    std::cout << "==done==" << std::endl;
    std::cout << "Total compute time was \t" << std::right << std::setw(12) << std::setprecision(5)
              << total_compute_time
              << "s" << std::endl;

    std::cout << "Reynolds number:  \t" << std::right << std::setw(12) << std::setprecision(12) << std::scientific
              << lbm::reynoldsNumber(*params, av_vels[params->maxIters - 1]) << std::endl;


    cout << "Now doing 5 runs and averaging IPU-reported timing:" << std::endl;
    unsigned long ipuTimer;
    double clockCycles = 0.;
    for (auto run = 0u; run < 5u; run++) {
        engine.run(1);
        engine.readTensor("readTimer", &ipuTimer);
        clockCycles += ipuTimer;
    }
    double clockFreq = device->getTarget().getTileClockFrequency();
    std::cout << "IPU reports " <<  std::fixed << clockFreq * 1e-6 << "MHz clock frequency" << std::endl;
    std::cout << "Average IPU timing for program is: " << std::fixed << std::setprecision(5) << std::setw(12)
              << clockCycles / 5.0 / clockFreq << "s" << std::endl;

    return EXIT_SUCCESS;
}
