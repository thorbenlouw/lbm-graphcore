
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
#include <poplar/Engine.hpp>

#include <sstream>

using namespace stencil;


int main(int argc, char *argv[]) {
    std::string inputFilename, outputFilename;
    unsigned numIters = 1u;
    unsigned numIpus = 1u;
    unsigned minRowsPerTile = 6u;
    unsigned minColsPerTile = 6u;
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
            ("min-rows-per-tile", "Min rows per tile (default 6)",
             cxxopts::value<unsigned>(minRowsPerTile)->default_value(std::to_string(grids::DefaultMinRowsPerTile)))
            ("min-cols-per-tile", "Min cols per tile (default 6)",
             cxxopts::value<unsigned>(minColsPerTile)->default_value(std::to_string(grids::DefaultMinColsPerTile)))
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
    auto fImage = zeroPad(toFloatImage(*maybeImg));
    auto img = fImage.intensities.data();

    auto device = useIpuModel ? utils::getIpuModel(numIpus) : utils::getIpuDevice(numIpus);

    std::cout << "Building graph";
    auto tic = std::chrono::high_resolution_clock::now();
    auto graph = poplar::Graph(*device);
    graph.addCodelets("StencilCodelet.cpp");

    const auto inImg = graph.addHostToDeviceFIFO(">>img", FLOAT,
                                                 NumChannels * fImage.height * fImage.width);
    const auto outImg = graph.addDeviceToHostFIFO("<<img", FLOAT,
                                                  NumChannels * fImage.height * fImage.width);

    auto imgTensor = graph.addVariable(FLOAT, {fImage.height, fImage.width, NumChannels}, "img");
    auto tmpImgTensor = graph.addVariable(FLOAT, {fImage.height, fImage.width, NumChannels}, "tmpImg");

    auto ipuLevelMappings = grids::partitionForIpus({fImage.height, fImage.width}, numIpus, 2000 * 1400);
    if (!ipuLevelMappings.has_value()) {
        std::cerr << "Couldn't fit the problem on the " << numIpus << " ipus." << std::endl;
        return EXIT_FAILURE;
    }
    auto tileLevelMappings = grids::toTilePartitions(*ipuLevelMappings, graph.getTarget().getTilesPerIPU(), minRowsPerTile,
                                                     minColsPerTile);
    auto workerLevelMappings = grids::toWorkerPartitions(tileLevelMappings);
    grids::serializeToJson(tileLevelMappings, "partitions.json");
    for (const auto &[target, slice]: tileLevelMappings) {
        graph.setTileMapping(utils::applySlice(imgTensor, slice), target.virtualTile());
        graph.setTileMapping(utils::applySlice(tmpImgTensor, slice), target.virtualTile());
    }

    auto inToOut = graph.addComputeSet("inToOut");
    auto outToIn = graph.addComputeSet("outToIn");
    auto zeros = std::vector<float>(std::max(fImage.width, fImage.height), 0.f);
    for (const auto &[target, slice]: workerLevelMappings) {
        // Halos top-left = (0,0) and no wraparound
        const auto halos = grids::Halos::forSliceTopIs0NoWrap(slice, {fImage.height, fImage.width});

        const auto maybeZerosVector = std::optional<Tensor>{};
        const auto maybeZeroScalar = std::optional<Tensor>{};

        const auto _target = target;
        const auto applyOrZero = [&graph, &_target, &zeros](const std::optional<grids::Slice2D> maybeSlice,
                                                            Tensor &tensor,
                                                            const unsigned borderSize = 1u) -> Tensor {
            if (maybeSlice.has_value()) {
                return utils::applySlice(tensor, *maybeSlice);
            } else {
                const auto zerosVector = graph.addConstant(FLOAT, {borderSize * 4}, zeros.data(), "{0...}");
                graph.setTileMapping(zerosVector, _target.virtualTile());
                return zerosVector;
            }
        };


        auto n = applyOrZero(halos.top, imgTensor, slice.width());
        auto s = applyOrZero(halos.bottom, imgTensor, slice.width());
        auto e = applyOrZero(halos.right, imgTensor, slice.height());
        auto w = applyOrZero(halos.left, imgTensor, slice.width());
        auto nw = applyOrZero(halos.topLeft, imgTensor);
        auto ne = applyOrZero(halos.topRight, imgTensor);
        auto sw = applyOrZero(halos.bottomLeft, imgTensor);
        auto se = applyOrZero(halos.bottomRight, imgTensor);
        const auto _slice = slice;
        const auto vertexName = [_slice]() -> std::string {
            if (_slice.height() > 1 && _slice.width() > 1) return "GaussianBlurCodelet<float>";
            else if (_slice.height() > 1) return "GaussianNarrow1ColBlurCodelet<float>";
            else if (_slice.width() > 1) return "GaussianWide1RowBlurCodelet<float>";
            else return "GaussianBlur1x1Codelet<float>";
        };
        auto v = graph.addVertex(inToOut,
                                 vertexName(),
                                 {
                                         {"width",  slice.width()},
                                         {"height", slice.height()},
                                         {"in",     utils::applySlice(imgTensor, slice)},
                                         {"out",    utils::applySlice(tmpImgTensor, slice)},
                                         {"n",      n},
                                         {"s",      s},
                                         {"e",      e},
                                         {"w",      w},
                                         {"nw",     nw},
                                         {"sw",     sw},
                                         {"ne",     ne},
                                         {"se",     se},
                                 }
        );
        graph.setCycleEstimate(v, 100);
        graph.setTileMapping(v, target.virtualTile());
        n = applyOrZero(halos.top, tmpImgTensor, slice.width());
        s = applyOrZero(halos.bottom, tmpImgTensor, slice.width());
        e = applyOrZero(halos.right, tmpImgTensor, slice.height());
        w = applyOrZero(halos.left, tmpImgTensor, slice.height());
        nw = applyOrZero(halos.topLeft, tmpImgTensor);
        ne = applyOrZero(halos.topRight, tmpImgTensor);
        sw = applyOrZero(halos.bottomLeft, tmpImgTensor);
        se = applyOrZero(halos.bottomRight, tmpImgTensor);
        v = graph.addVertex(outToIn,
                            vertexName(),
                            {
                                    {"width",  slice.width()},
                                    {"height", slice.height()},
                                    {"out",    utils::applySlice(imgTensor, slice)},
                                    {"in",     utils::applySlice(tmpImgTensor, slice)},
                                    {"n",      n},
                                    {"s",      s},
                                    {"e",      e},
                                    {"w",      w},
                                    {"nw",     nw},
                                    {"sw",     sw},
                                    {"ne",     ne},
                                    {"se",     se},
                            }
        );
        graph.setCycleEstimate(v, 100);
        graph.setTileMapping(v, target.virtualTile());
    }
    Sequence stencilProgram = {Execute(inToOut), Execute(outToIn)};


    auto copyToDevice = Copy(inImg, imgTensor);
    auto copyBackToHost = Copy(imgTensor, outImg);

    const auto programs = std::vector<Program>{copyToDevice,
                                               Repeat(numIters, stencilProgram),
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

            engine.printProfileSummary(std::cout,
                                       OptionFlags{{"showExecutionSteps", "false"}});
        }
    }


    auto cImg = toCharImage(stripPadding(fImage));

    if (!savePng(cImg, outputFilename)) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
