#include <cstdlib>
#include <cxxopts.hpp>
#include <lodepng.h>
#include "StructuredGridUtils.hpp"
#include <chrono>
#include "StencilUtils.hpp"
#include "GraphcoreUtils.hpp"
#include <poplar/IPUModel.hpp>
#include <popops/codelets.hpp>
#include <popops/Zero.hpp>
#include <iostream>
#include <poplar/Program.hpp>
#include <poplin/Convolution.hpp>
#include <poplin/codelets.hpp>
#include <sstream>

using namespace stencil;
using namespace poplar;
using namespace poplar::program;
using namespace poplin;

int main(int argc, char *argv[]) {
    std::string inputFilename, outputFilename;
    unsigned numIters = 1u;
    unsigned numIpus = 1u;
    bool compileOnly = false;
    bool debug = false;
    bool useIpuModel = false;

    cxxopts::Options options(argv[0],
                             " - Runs a 3x3 Gaussian Blur stencil over an input image a number of times using Poplibs on the IPU");
    options.add_options()
            ("n,num-iters", "Number of iterations", cxxopts::value<unsigned>(numIters)->default_value("1"))
            ("i,image", "filename input image (must be a 4-channel PNG image)",
             cxxopts::value<std::string>(inputFilename))
            ("num-ipus", "Number of IPUs to target (1,2,4,8 or 16)", cxxopts::value<unsigned>(numIpus))
            ("o,output", "filename output (blurred image)", cxxopts::value<std::string>(outputFilename))
            ("d,debug", "Run in debug mode (capture profiling information)")
            ("compile-only", "Only compile the graph and write to stencil_<width>x<height>.exe, don't run")
            ("m,ipu-model", "Run on IPU model (emulator) instead of real device");

    try {
        auto opts = options.parse(argc, argv);
        debug = opts["debug"].as<bool>();
        compileOnly = opts["compile-only"].as<bool>();
        useIpuModel = opts["ipu-model"].as<bool>();
        if (opts.count("n") + opts.count("i") + opts.count("num-ipus") + opts.count("o") < 0) {
            std::cerr << options.help() << std::endl;
            return EXIT_FAILURE;
        }
    } catch (cxxopts::OptionParseException &) {
        std::cerr << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    auto maybeImg = loadPng(inputFilename);
    if (!maybeImg.has_value()) {
        return EXIT_FAILURE;
    }

    cout << inputFilename << " is " << maybeImg->width << "x" << maybeImg->height << " pixels in size." << std::endl;

    auto tmp_img = std::make_unique<float[]>((maybeImg->width) * (maybeImg->height) * NumChannels);
    auto fImage = toChannelsFirst(toFloatImage(*maybeImg));
    auto img = fImage.intensities.data();

    auto device = useIpuModel ? utils::getIpuModel(numIpus) : utils::getIpuDevice(numIpus);


    std::cout << "Building graph";
    auto tic = std::chrono::high_resolution_clock::now();
    auto graph = poplar::Graph(*device);
    poplin::addCodelets(graph);
    popops::addCodelets(graph);
    auto kernel = ArrayRef{1.f / 16, 2.f / 16, 1.f / 16,
                           2.f / 16, 4.f / 16, 2.f / 16,
                           1.f / 16, 2.f / 16, 1.f / 16};
    auto gaussianBlurMask = graph.addConstant(FLOAT, {3, 3},
                                              kernel,
                                              "3x3gaussian_blur_kernel");
    graph.setTileMapping(gaussianBlurMask, 0);
    auto padInput = poplin::ConvParams::InputTransform{{0,     0},
                                                       {0,     0},
                                                       {1,     1},
                                                       {1,     1},
                                                       {1,     1},
                                                       {false, false}};
    auto transformKernel = poplin::ConvParams::InputTransform{
            {0,     0},
            {0,     0},
            {1,     1},
            {0,     0},
            {0,     0},
            {false, false}};

    auto transformOutput = poplin::ConvParams::OutputTransform{
            {0, 0},
            {0, 0},
            {1, 1},
            {0, 0},
            {0, 0}};

    auto convParams = ConvParams(FLOAT, FLOAT, 1, {fImage.height, fImage.width},
                                 {3, 3}, 4, 4, 1,
                                 padInput, transformKernel,
                                 transformOutput);

    auto convOptions = OptionFlags{{"pass",                  "INFERENCE_FWD"},
                                   {"remapOutputTensor",     "true"},
                                   {"use128BitConvUnitLoad", "true"}};
    auto imgTensor = createInput(graph, convParams, "convInput", convOptions);
    // imgTensor is 1x4x3x3
    auto weights = createWeights(graph, convParams, "convWeights", convOptions);
    // weights is 1x4x4x3x3

    Sequence copyKernelToWeights;
    popops::zero(graph, weights, copyKernelToWeights, "zeroWeights");
    for (auto outChans = 0u; outChans < NumChannels; outChans++) {
        for (auto inChans = 0u; inChans < NumChannels; inChans++) {
            if (inChans == outChans) copyKernelToWeights.add(Copy(gaussianBlurMask, weights[0][outChans][inChans]));
        }
    };

    const auto inImg = graph.addHostToDeviceFIFO(">>img", FLOAT,
                                                 NumChannels * fImage.height * fImage.width);
    const auto outImg = graph.addDeviceToHostFIFO("<<img", FLOAT,
                                                  NumChannels * fImage.height * fImage.width);
    Sequence stencilProgram;
    auto out = convolution(graph, imgTensor, weights, convParams, false,
                           stencilProgram, "convolution", convOptions
    );
    stencilProgram.add(Copy(out, imgTensor));

    auto copyToDevice = Copy(inImg, imgTensor);
    auto copyBackToHost = Copy(imgTensor, outImg);

    const auto programs = std::vector<Program>{copyToDevice,
                                               Sequence{copyKernelToWeights, Repeat(2 * numIters, stencilProgram)},
                                               copyBackToHost};


    auto toc = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic).count();
    std::cout << " took " << std::right << std::setw(12) << std::setprecision(5) << diff << "s" <<
              std::endl;

    if (debug) {
        utils::serializeGraph(graph);
    }
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
    std::cout << "Compiling graph";
    tic = std::chrono::high_resolution_clock::now();
    if (compileOnly) {
        auto exe = poplar::compileGraph(graph, programs, debug ? utils::POPLAR_ENGINE_OPTIONS_DEBUG
                                                               : utils::POPLAR_ENGINE_OPTIONS_NODEBUG);
        toc = std::chrono::high_resolution_clock::now();
        diff = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic).count();
        std::cout << " took " << std::right << std::setw(12) << std::setprecision(5) << diff << "s" <<
                  std::endl;

        stringstream filename;
        filename << "stencil_" << maybeImg->width << "x" << maybeImg->height << ".exe";
        ofstream exe_file;
        exe_file.open(filename.str());
        exe.serialize(exe_file);
        exe_file.close();

        return EXIT_SUCCESS;
    } else {
        auto engine = Engine(graph, programs,
                             utils::POPLAR_ENGINE_OPTIONS_DEBUG);

        toc = std::chrono::high_resolution_clock::now();
        diff = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic).count();
        std::cout << " took " << std::right << std::setw(12) << std::setprecision(5) << diff << "s" <<
                  std::endl;


        engine.connectStream(">>img", img);
        engine.connectStream("<<img", img);

        engine.load(*device);

        utils::timedStep("Copying to device", [&]() -> void {
            engine.run(0);
        });

        utils::timedStep("Running stencil iterations", [&]() -> void {
            engine.run(1);
        });
        utils::timedStep("Copying to host", [&]() -> void {
            engine.run(2);
        });

        if (debug) {
            utils::captureProfileInfo(engine);

//            engine.printProfileSummary(std::cout,
//                                       OptionFlags{{"showExecutionSteps", "false"}});
        }
    }


    auto cImg = toCharImage(toChannelsLast(fImage));

    if (!savePng(cImg, outputFilename)) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}