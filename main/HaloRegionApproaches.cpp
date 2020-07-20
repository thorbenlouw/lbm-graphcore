

#include <cstdlib>
#include <cxxopts.hpp>
#include "StructuredGridUtils.hpp"
#include <chrono>
#include "GraphcoreUtils.hpp"
#include <poplar/IPUModel.hpp>
#include <popops/Zero.hpp>
#include <popops/codelets.hpp>
#include <iostream>
#include <poplar/Program.hpp>

#include <sstream>

//constexpr auto NumTilesInIpuCol = 2u;
constexpr auto NumTilesInIpuCol = 32u;
constexpr auto NumTiles = 1216u;

auto fill(Graph &graph, const Tensor &tensor, const float value, const unsigned tileNumber, ComputeSet &cs) -> void {
    auto v = graph.addVertex(cs,
                             "Fill<float>",
                             {
                                     {"result", tensor.flatten()},
                                     {"val",    value}
                             }
    );
    graph.setCycleEstimate(v, tensor.numElements());
    graph.setTileMapping(v, tileNumber);
}

auto implicitStrategy(Graph &graph, vector<Program> &vector, const unsigned numTiles,
                      const unsigned blockSizePerTile, const unsigned numIters) -> std::vector<Program> {
    const auto NumTilesInIpuRow = numTiles / NumTilesInIpuCol;

    auto in = graph.addVariable(FLOAT, {NumTilesInIpuRow * blockSizePerTile, NumTilesInIpuCol * blockSizePerTile},
                                "in");
    auto out = graph.addVariable(FLOAT, {NumTilesInIpuRow * blockSizePerTile, NumTilesInIpuCol * blockSizePerTile},
                                 "out");

    // Place the blocks of in and out on the right tiles
    auto initCs = graph.addComputeSet("init");
    for (auto tile = 0u; tile < numTiles; tile++) {
        auto ipuRow = tile / NumTilesInIpuCol;
        auto ipuCol = tile % NumTilesInIpuCol;
        auto startRowInTensor = ipuRow * blockSizePerTile;
        auto endRowInTensor = startRowInTensor + blockSizePerTile;
        auto startColInTensor = ipuCol * blockSizePerTile;
        auto endColInTensor = startColInTensor + blockSizePerTile;
        auto block = [=](const Tensor &t) -> Tensor {
            return t.slice({startRowInTensor, startColInTensor}, {endRowInTensor, endColInTensor});
        };
        graph.setTileMapping(block(in), tile);
        graph.setTileMapping(block(out), tile);
        fill(graph, block(in), (float) tile, tile, initCs);
    }

    auto stencilProgram = [&]() -> Program {
        ComputeSet compute1 = graph.addComputeSet("implicitCompute1");
        ComputeSet compute2 = graph.addComputeSet("implicitCompute2");
        const auto numTiles = NumTiles;//graph.getTarget().getNumTiles();;
        for (auto tile = 0u; tile < numTiles; tile++) {
            auto ipuRow = tile / NumTilesInIpuCol;
            auto ipuCol = tile % NumTilesInIpuCol;
            auto NumTilesInIpuRow = numTiles / NumTilesInIpuCol;

            auto v = graph.addVertex(compute1,
                                     "IncludedHalosApproach<float>",
                                     {
                                             {"in",  in[tile]},
                                             {"out", out[tile]}
                                             //    {"n", ipuRow > 0 ? in[tile-NumTilesInIpuCol][blockSizePerTile-1] : } // need to feed in n,s,w,e etc.
                                     }
            );
            graph.setCycleEstimate(v, 9);
            graph.setTileMapping(v, tile);
        }
        return Sequence(Execute(compute1), Execute(compute2));
    };


    return {Execute(initCs), Repeat{numIters, stencilProgram()}};

}

auto explicitManyTensorStrategy(Graph &graph, vector<Program> &vector, const unsigned numTiles,
                                const unsigned blockSizePerTile, const unsigned numIters) -> std::vector<Program> {
    const auto NumTilesInIpuRow = numTiles / NumTilesInIpuCol;

    // Place the blocks of in and out on the right tiles

    auto blocksForIncludedHalosIn = std::vector<Tensor>{numTiles};
    auto blocksForIncludedHalosOut = std::vector<Tensor>{numTiles};

    auto initialiseProgram = Sequence{};
    auto initialiseCs = graph.addComputeSet("init");

    for (auto tile = 0u; tile < numTiles; tile++) {
        auto ipuRow = tile / NumTilesInIpuCol;
        auto ipuCol = tile % NumTilesInIpuCol;

        blocksForIncludedHalosIn[tile] = graph.addVariable(FLOAT, {blockSizePerTile + 2, blockSizePerTile + 2},
                                                           "in" + std::to_string(tile));
        blocksForIncludedHalosOut[tile] = graph.addVariable(FLOAT, {blockSizePerTile + 2, blockSizePerTile + 2},
                                                            "in" + std::to_string(tile));
        graph.setTileMapping(blocksForIncludedHalosIn[tile], tile);
        graph.setTileMapping(blocksForIncludedHalosOut[tile], tile);
        fill(graph, blocksForIncludedHalosIn[tile], (float) tile, tile, initialiseCs);
        fill(graph, blocksForIncludedHalosOut[tile], (float) tile, tile, initialiseCs);
        //  zero out the tlbr grids' halos appropriately
        if (ipuRow == 0) {
            popops::zero(graph, blocksForIncludedHalosIn[tile][0], initialiseProgram, "zeroTopHaloEdge");
        }
        if (ipuRow == NumTilesInIpuRow - 1) {
            popops::zero(graph, blocksForIncludedHalosIn[tile][blockSizePerTile + 1], initialiseProgram,
                         "zeroBottomEdge");
        }
        if (ipuCol == 0) {
            popops::zero(graph, blocksForIncludedHalosIn[tile].slice({0, 0}, {blockSizePerTile + 2, 1}),
                         initialiseProgram,
                         "zeroLeftHaloEdge");
        }
        if (ipuCol == NumTilesInIpuCol - 1) {
            popops::zero(graph, blocksForIncludedHalosIn[tile].slice({0, blockSizePerTile + 1},
                                                                     {blockSizePerTile + 2, blockSizePerTile + 2}),
                         initialiseProgram, "zeroRightHaloEdge");
        }
    }

    auto stencilProgram = [&]() -> Sequence {
        ComputeSet compute1 = graph.addComputeSet("explicitCompute1");
        ComputeSet compute2 = graph.addComputeSet("explicitCompute2");
        const auto numTiles = NumTiles;// graph.getTarget().getNumTiles();
        auto NumTilesInIpuRow = numTiles / NumTilesInIpuCol;

        const auto haloExchangeFn = [&](std::vector<Tensor> &t) -> Sequence {
            auto northSouthWave = Sequence{};
            auto leftRightWave = Sequence{};
            for (auto tile = 0u; tile < numTiles; tile++) {
                auto ipuRow = tile / NumTilesInIpuCol;
                auto ipuCol = tile % NumTilesInIpuCol;
                // copy my top row upwards (minus first 2 elems)
                if (ipuRow > 0) {
                    northSouthWave.add(Copy(t[tile][1].slice(2, blockSizePerTile + 2),
                                            t[tile - NumTilesInIpuCol][blockSizePerTile + 1].slice(2, blockSizePerTile +
                                                                                                      2)));
                }
                // copy my bottom row downwards (minus first 2 elems)
                if (ipuRow < NumTilesInIpuRow - 1) {
                    northSouthWave.add(Copy(t[tile][blockSizePerTile].slice(2, blockSizePerTile + 2),
                                            t[tile + NumTilesInIpuCol][0].slice(2, blockSizePerTile + 2)));
                }

                // copy my right row rightwards (including all elems)
                if (ipuCol < NumTilesInIpuCol - 1) {
                    leftRightWave.add(
                            Copy(t[tile].slice({0, blockSizePerTile}, {blockSizePerTile + 2, blockSizePerTile + 1}),
                                 t[tile + 1].slice({0, 0}, {blockSizePerTile + 2, 1})));
                }
                // copy my left row leftwards (including all elems)
                if (ipuCol > 0) {
                    leftRightWave.add(Copy(t[tile].slice({0, 0}, {blockSizePerTile + 2, 1}),
                                           t[tile - 1].slice({0, blockSizePerTile + 1},
                                                             {blockSizePerTile + 2, blockSizePerTile + 2})));
                }
            }
            return Sequence(northSouthWave, leftRightWave);
        };

        auto haloExchange1 = haloExchangeFn(blocksForIncludedHalosIn);
        auto haloExchange2 = haloExchangeFn(blocksForIncludedHalosOut);

        for (auto tile = 0u; tile < numTiles; tile++) {
            auto v = graph.addVertex(compute1,
                                     "IncludedHalosApproach<float>",
                                     {
                                             {"in",  blocksForIncludedHalosIn[tile]},
                                             {"out", blocksForIncludedHalosOut[tile]},
                                     }
            );
            graph.setCycleEstimate(v, blockSizePerTile * blockSizePerTile * 9);
            graph.setTileMapping(v, tile);
            v = graph.addVertex(compute2,
                                "IncludedHalosApproach<float>",
                                {
                                        {"in",  blocksForIncludedHalosOut[tile]},
                                        {"out", blocksForIncludedHalosIn[tile]},
                                }
            );
            graph.setCycleEstimate(v, blockSizePerTile * blockSizePerTile * 9);
            graph.setTileMapping(v, tile);
        }


        return Sequence(haloExchange1, Execute(compute1), haloExchange2, Execute(compute2));
    };
    return {Sequence{initialiseProgram, Execute(initialiseCs)}, Repeat{numIters, stencilProgram()}};

}

auto explicitOneTensorStrategy(Graph &graph, vector<Program> &vector, const unsigned numTiles,
                               const unsigned blockSizePerTile, const unsigned numIters) -> std::vector<Program> {
    const auto NumTilesInIpuRow = numTiles / NumTilesInIpuCol;

    auto expandedIn = graph.addVariable(FLOAT,
                                        {NumTilesInIpuRow * (blockSizePerTile + 2),
                                         NumTilesInIpuCol * (blockSizePerTile + 2)},
                                        "expandedIn");
    auto expandedOut = graph.addVariable(FLOAT,
                                         {NumTilesInIpuRow * (blockSizePerTile + 2),
                                          NumTilesInIpuCol * (blockSizePerTile + 2)},
                                         "expandedOut");


    auto initialiseProgram = Sequence{};
    auto initialiseCs = graph.addComputeSet("init");

    for (auto tile = 0u; tile < numTiles; tile++) {
        auto ipuRow = tile / NumTilesInIpuCol;
        auto ipuCol = tile % NumTilesInIpuCol;

        const auto blockWithHalo = [&](const Tensor &t) -> Tensor {
            const auto startRow = ipuRow * (blockSizePerTile + 2);
            const auto startCol = ipuCol * (blockSizePerTile + 2);
            return t.slice({startRow, startCol},
                           {startRow + blockSizePerTile + 2, startCol + blockSizePerTile + 2});
        };
        graph.setTileMapping(blockWithHalo(expandedIn), tile);
        graph.setTileMapping(blockWithHalo(expandedOut), tile);

    }
    popops::zero(graph, expandedIn, initialiseProgram);
    popops::zero(graph, expandedOut, initialiseProgram);

    for (auto tile = 0u; tile < numTiles; tile++) {
        auto ipuRow = tile / NumTilesInIpuCol;
        auto ipuCol = tile % NumTilesInIpuCol;

        const auto block = [&](const Tensor &t) -> Tensor {
            const auto startRow = ipuRow * (blockSizePerTile + 2) + 1;
            const auto startCol = ipuCol * (blockSizePerTile + 2) + 1;
            return t.slice({startRow, startCol}, {startRow + blockSizePerTile, startCol + blockSizePerTile});
        };
        fill(graph, block(expandedIn), (float) tile, tile, initialiseCs);
        fill(graph, block(expandedOut), (float) tile, tile, initialiseCs);
    }

    auto stencilProgram = [&]() -> Sequence {
        ComputeSet compute1 = graph.addComputeSet("explicitCompute1");
        ComputeSet compute2 = graph.addComputeSet("explicitCompute2");
        const auto numTiles = NumTiles;// graph.getTarget().getNumTiles();
        auto NumTilesInIpuRow = numTiles / NumTilesInIpuCol;

        const auto haloExchangeFn = [&](Tensor &t) -> Sequence {
            auto northSouthWave = Sequence{};
            auto leftRightWave = Sequence{};
            for (auto tile = 0u; tile < numTiles; tile++) {
                auto ipuRow = tile / NumTilesInIpuCol;
                auto ipuCol = tile % NumTilesInIpuCol;

                const auto topHaloRow = ipuRow * (blockSizePerTile + 2);
                const auto bottomHaloRow = topHaloRow + blockSizePerTile + 1;
                const auto leftHaloCol = ipuCol * (blockSizePerTile + 2);
                const auto rightHaloCol = leftHaloCol+blockSizePerTile + 1;


                // copy my top row upwards (minus first 2 elems)
                if (ipuRow > 0) {
                    northSouthWave.add(
                            Copy(t.slice({topHaloRow + 1, leftHaloCol + 2}, {topHaloRow + 2, rightHaloCol + 1}),
                                 t.slice({topHaloRow - 1, leftHaloCol + 2}, {topHaloRow, rightHaloCol + 1})));
                }
                // copy my bottom row downwards (minus first 2 elems)
                if (ipuRow < NumTilesInIpuRow - 1) {
                    northSouthWave.add(
                            Copy(t.slice({bottomHaloRow - 1, leftHaloCol + 2}, {bottomHaloRow, rightHaloCol + 1}),
                                 t.slice({bottomHaloRow + 1, leftHaloCol + 2}, {bottomHaloRow + 2, rightHaloCol + 1})));
                }

                // copy my right row rightwards (including all elems)
                if (ipuCol < NumTilesInIpuCol - 1) {
                    leftRightWave.add(
                            Copy(t.slice({topHaloRow, rightHaloCol - 1}, {bottomHaloRow + 1, rightHaloCol}),
                                 t.slice({topHaloRow, rightHaloCol + 1}, {bottomHaloRow + 1, rightHaloCol + 2})));
                }
                // copy my left row leftwards (including all elems)
                if (ipuCol > 0) {
                    leftRightWave.add(Copy(t.slice({topHaloRow, leftHaloCol + 1}, {bottomHaloRow + 1, leftHaloCol + 2}),
                                           t.slice({topHaloRow, leftHaloCol - 1}, {bottomHaloRow + 1, leftHaloCol})));
                }
            }
            return Sequence(northSouthWave, leftRightWave);
        };

        auto haloExchange1 = haloExchangeFn(expandedIn);
        auto haloExchange2 = haloExchangeFn(expandedOut);

        for (auto tile = 0u; tile < numTiles; tile++) {
            auto ipuRow = tile / NumTilesInIpuCol;
            auto ipuCol = tile % NumTilesInIpuCol;

            const auto topHaloRow = ipuRow * (blockSizePerTile + 2);
            const auto bottomHaloRow = topHaloRow + blockSizePerTile + 1;
            const auto leftHaloCol = ipuCol * (blockSizePerTile + 2);
            const auto rightHaloCol = leftHaloCol+blockSizePerTile + 1;

            const auto block = [&](const Tensor &t) -> Tensor {
                return t.slice({topHaloRow, leftHaloCol}, {bottomHaloRow + 1, rightHaloCol + 1});
            };
            auto v = graph.addVertex(compute1,
                                     "IncludedHalosApproach<float>",
                                     {
                                             {"in",  block(expandedIn)},
                                             {"out", block(expandedOut)},
                                     }
            );
            graph.setCycleEstimate(v, blockSizePerTile * blockSizePerTile * 9);
            graph.setTileMapping(v, tile);
            v = graph.addVertex(compute2,
                                "IncludedHalosApproach<float>",
                                {
                                        {"out", block(expandedIn)},
                                        {"in",  block(expandedOut)},
                                }
            );
            graph.setCycleEstimate(v, blockSizePerTile * blockSizePerTile * 9);
            graph.setTileMapping(v, tile);
        }


        return Sequence(haloExchange1, Execute(compute1), haloExchange2, Execute(compute2));
    };
    return {Sequence{initialiseProgram, Execute(initialiseCs)}, Repeat{numIters, stencilProgram()}};
}

int main(int argc, char *argv[]) {
    unsigned numIters = 1u;
    unsigned numIpus = 1u;
    unsigned blockSizePerTile = 100;
    std::string strategy = "implicit";
    bool compileOnly = false;
    bool debug = false;
    bool useIpuModel = false;

    cxxopts::Options options(argv[0],
                             " - Prints timing for a run of a simple Moore neighbourhood average stencil ");
    options.add_options()
            ("h,halo-exhange-strategy", "{implicit,explicitManyTensors,explicitOneTensor}",
             cxxopts::value<std::string>(strategy)->default_value("implicit"))
            ("n,num-iters", "Number of iterations", cxxopts::value<unsigned>(numIters)->default_value("1"))
            ("b,block-size", "Block size per Tile",
             cxxopts::value<unsigned>(blockSizePerTile)->default_value("100"))
            ("num-ipus", "Number of IPUs to target (1,2,4,8 or 16)",
             cxxopts::value<unsigned>(numIpus)->default_value("1"))
            ("d,debug", "Run in debug mode (capture profiling information)")
            ("compile-only", "Only compile the graph and write to stencil_<width>x<height>.exe, don't run")
            ("m,ipu-model", "Run on IPU model (emulator) instead of real device");

    try {
        auto opts = options.parse(argc, argv);
        debug = opts["debug"].as<bool>();
        compileOnly = opts["compile-only"].as<bool>();
        useIpuModel = opts["ipu-model"].as<bool>();
        if (opts.count("n") + opts.count("b") < 2) {
            std::cerr << options.help() << std::endl;
            return EXIT_FAILURE;
        }
        if (!(strategy == "implicit" || strategy == "explicitManyTensors" || strategy == "explicitOneTensor")) {
            std::cerr << options.help() << std::endl;
            return EXIT_FAILURE;
        }
    } catch (cxxopts::OptionParseException &) {
        std::cerr << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    auto device = useIpuModel ? utils::getIpuModel(numIpus) : utils::getIpuDevice(numIpus);
    if (!device.has_value()) {
        return EXIT_FAILURE;
    }

    std::cout << "Building graph";
    auto tic = std::chrono::high_resolution_clock::now();
    auto graph = poplar::Graph(*device);
    const auto numTiles = NumTiles;//graph.getTarget().getNumTiles();;
    auto NumTilesInIpuRow = numTiles / NumTilesInIpuCol;
    assert(NumTilesInIpuCol * NumTilesInIpuRow == numTiles);

    graph.addCodelets("HaloRegionApproachesCodelets.cpp");
    popops::addCodelets(graph);


    auto programs = std::vector<Program>{};
    if (strategy == "implicit") {
        programs = implicitStrategy(graph, programs, numTiles, blockSizePerTile, numIters);
    } else if (strategy == "explicitManyTensors") {
        programs = explicitManyTensorStrategy(graph, programs, numTiles, blockSizePerTile, numIters);
    } else if (strategy == "explicitOneTensor") {
        programs = explicitOneTensorStrategy(graph, programs, numTiles, blockSizePerTile, numIters);
    } else {
        return EXIT_FAILURE;
    }


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

        const auto filename = "graph.exe";
        ofstream exe_file;
        exe_file.open(filename);
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

        engine.load(*device);

        engine.run(0);

        utils::timedStep("Running explicit halo exchange iterations", [&]() -> void {
            engine.run(1);
        });


        if (debug) {
            utils::captureProfileInfo(engine);

//            engine.printProfileSummary(std::cout,
//                                       OptionFlags{{"showExecutionSteps", "false"}});
        }
    }

    return EXIT_SUCCESS;
}
