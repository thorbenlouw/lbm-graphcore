#include <cstdlib>
#include <cxxopts.hpp>
#include <lodepng.h>
#include "StructuredGridUtils.hpp"
#include <chrono>
#include "StencilUtils.hpp"
#include "GraphcoreUtils.hpp"
#include <poplar/IPUModel.hpp>
#include <popops/Reduce.hpp>
#include <popops/codelets.hpp>
#include <iostream>
#include <poplar/Program.hpp>
#include <poplin/Convolution.hpp>
#include <poplin/codelets.hpp>

using namespace stencil;
using namespace poplar;
using namespace poplar::program;
using namespace poplin;

int main(int argc, char *argv[]) {
    std::string inputFilename, outputFilename;
    unsigned numIters = 1u;
    unsigned numIpus = 1u;

    cxxopts::Options options(argv[0],
                             " - Runs a 3x3 Gaussian Blur stencil over an input image a number of times using Poplibs on the IPU");
    options.add_options()
            ("n,num-iters", "Number of iterations", cxxopts::value<unsigned>(numIters)->default_value("1"))
            ("i,image", "filename input image (must be a 4-channel PNG image)",
             cxxopts::value<std::string>(inputFilename))
            ("num-ipus", "Number of IPUs to target (1,2,4,8 or 16)", cxxopts::value<unsigned>(numIpus))
            ("o,output", "filename output (blurred image)", cxxopts::value<std::string>(outputFilename));
    auto opts = options.parse(argc, argv);

    auto maybeImg = loadPng(inputFilename);
    if (!maybeImg.has_value()) {
        return EXIT_FAILURE;
    }

    cout << inputFilename << " is " << maybeImg->width << "x" << maybeImg->height << " pixels in size." << std::endl;

    alignas(64) auto img = std::make_unique<float[]>((maybeImg->width + 2) * (maybeImg->height + 2) * NumChannels);
    alignas(64) auto tmp_img = std::make_unique<float[]>((maybeImg->width + 2) * (maybeImg->height + 2) * NumChannels);
    auto fImageDescr = toPaddedFloatImage(*maybeImg, img);

    auto device = utils::getIpuDevice(numIpus);
    //   auto device = utils::getIpuModel(numIpus);

    std::cout << "Building graph";
    auto tic = std::chrono::high_resolution_clock::now();
    auto graph = poplar::Graph(*device);
    poplin::addCodelets(graph);
    const auto mask = ArrayRef{1.f / 16, 2.f / 16, 1.f / 16,
                               2.f / 16, 4.f / 16, 2.f / 16,
                               1.f / 16, 2.f / 16, 1.f / 16};
    auto gaussianBlurMask = graph.addConstant(FLOAT, {3, 3},
                                              mask,
                                              "gaussian_blur_mask");
    graph.setTileMapping(gaussianBlurMask, 0);
    //auto imgTensor = graph.addVariable(FLOAT, {NumChannels, fImageDescr.height, fImageDescr.width}, "img");
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

    auto convParams = ConvParams(FLOAT, FLOAT, 1, {fImageDescr.height, fImageDescr.width},
                                 {3, 3}, 4, 4, 1,
                                 padInput, transformKernel,
                                 transformOutput);

    auto convOptions = OptionFlags{{"pass",                  "INFERENCE_FWD"},
//                                   {"enableFastReduce", "true"},
                                   {"remapOutputTensor",     "true"},
                                   {"use128BitConvUnitLoad", "true"}};
    auto imgTensor = createInput(graph, convParams, "convInput", convOptions);
    auto weights = createWeights(graph, convParams, "convWeights", convOptions);
    // weights is 1x4x3x3

    Sequence copyKernelToWeights;
    for (auto outChans = 0u; outChans < NumChannels; outChans++) {
        for (auto inChans = 0u; inChans < NumChannels; inChans++) {
            copyKernelToWeights.add(Copy(gaussianBlurMask, weights[0][outChans][inChans]));
        }
    };

    const auto inImg = graph.addHostToDeviceFIFO(">>img", FLOAT,
                                                 NumChannels * fImageDescr.height * fImageDescr.width);
    const auto outImg = graph.addDeviceToHostFIFO("<<img", FLOAT,
                                                  NumChannels * fImageDescr.height * fImageDescr.width);
    Sequence s;
    auto out = convolution(graph, imgTensor, weights, convParams, false,
                           s, "convolution", convOptions
    );
    auto copyTmpToImg = Copy(out, imgTensor);
    Program stencilProgram = Sequence{s, copyTmpToImg};

    auto copyToDevice = Copy(inImg, imgTensor);
    auto copyBackToHost = Copy(imgTensor, outImg);

    const auto progs = std::vector<Program>{copyToDevice,
                                            Sequence{copyKernelToWeights, Repeat(2 * numIters, stencilProgram)},
                                            copyBackToHost};
    auto toc = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic).count();
    std::cout << " took " << std::right << std::setw(12) << std::setprecision(5) << diff << "s" <<
              std::endl;

    std::cout << "Compiling graph";
    tic = std::chrono::high_resolution_clock::now();
    auto exe = poplar::compileGraph(graph, progs, utils::POPLAR_ENGINE_OPTIONS_DEBUG);

    auto engine = Engine(graph, progs,
                         utils::POPLAR_ENGINE_OPTIONS_DEBUG);

    toc = std::chrono::high_resolution_clock::now();
    diff = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic).count();
    std::cout << " took " << std::right << std::setw(12) << std::setprecision(5) << diff << "s" <<
              std::endl;

    engine.connectStream(">>img", img.get());
    engine.connectStream("<<img", img.get());

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

    utils::captureProfileInfo(engine);
//
//    engine.printProfileSummary(std::cout,
//                               OptionFlags{{"showExecutionSteps", "true"}});


    auto cImg = toUnpaddedCharsImage(fImageDescr, img);

    if (!
            savePng(cImg, outputFilename
            )) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
