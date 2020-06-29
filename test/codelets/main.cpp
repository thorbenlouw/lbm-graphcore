#include "gtest/gtest.h"

#include "../../main/GraphcoreUtils.hpp"
#include <poplar/Device.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Engine.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>

using namespace poplar;
using namespace poplar::program;

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}


TEST(averageVelocity, testMaskedSumPartial) {
    auto device = poplar::Device::createCPUDevice();
    auto graph = Graph{device.getTarget()};
    graph.addCodelets("D2Q9Codelets.cpp");
    auto tensors = lbm::TensorMap{};
    tensors["cells"] = graph.addVariable(FLOAT, {1, 3, 9}, "cells");
    graph.setTileMapping(tensors["cells"], 0);
    graph.setInitialValue(tensors["cells"], ArrayRef<float>{1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                            0, 1, 2, 3, 4, 5, 6, 7, 8,
                                                            2, 3, 4, 5, 6, 7, 8, 9, 10});
    tensors["obstacles"] = graph.addVariable(BOOL, {3}, "obstacles");
    graph.setTileMapping(tensors["obstacles"], 0);
    graph.setInitialValue(tensors["obstacles"], ArrayRef<bool>{true, false, true});

    tensors["result"] = graph.addVariable(FLOAT, {}, "result");
    graph.setTileMapping(tensors["result"], 0);
    graph.setInitialValue(tensors["result"], 0);

    graph.createHostRead("readResult", tensors["result"], false);


    auto cs = graph.addComputeSet("test");

    auto v = graph.addVertex(cs,
                             "MaskedSumPartial",
                             {
                                     {"cells",     tensors["cells"].flatten()},
                                     {"obstacles", tensors["obstacles"].flatten()},
                                     {"numCells",  9},
                                     {"out",       tensors["result"]},
                             });

    graph.setCycleEstimate(v, 1);
    graph.setTileMapping(v, 0);

    auto engine = lbm::createDebugEngine(graph, {Execute(cs)});
    engine.load(device);
    engine.run();

    float result = 0;
    engine.readTensor("readResult", &result);

    ASSERT_EQ(result, (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9) + (2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10));


}


TEST(averageVelocity, testReducePartials) {
    auto device = poplar::Device::createCPUDevice();
    auto graph = Graph{device.getTarget()};
    graph.addCodelets("D2Q9Codelets.cpp");
    auto tensors = lbm::TensorMap{};
    tensors["partials"] = graph.addVariable(FLOAT, {3}, "partials");
    graph.setTileMapping(tensors["partials"], 0);
    graph.setInitialValue(tensors["partials"], ArrayRef<float>{1, 2, 3});
    tensors["result"] = graph.addVariable(FLOAT, {}, "result");
    graph.setTileMapping(tensors["result"], 0);


    graph.createHostRead("readResult", tensors["result"], false);


    auto cs = graph.addComputeSet("test");

    auto v = graph.addVertex(cs,
                             "ReducePartials",
                             {
                                     {"partials",    tensors["partials"].flatten()},
                                     {"numPartials", 3},
                                     {"out",         tensors["result"]},
                             });

    graph.setCycleEstimate(v, 1);
    graph.setTileMapping(v, 0);

    auto engine = lbm::createDebugEngine(graph, {Execute(cs)});
    engine.load(device);
    engine.run();

    float result = 0;
    engine.readTensor("readResult", &result);

    ASSERT_EQ(result, 6);


}


TEST(averageVelocity, testAppendReducedSum) {
    auto device = poplar::Device::createCPUDevice();
    auto graph = Graph{device.getTarget()};
    graph.addCodelets("D2Q9Codelets.cpp");
    popops::addCodelets(graph);

    auto tensors = lbm::TensorMap{};
    tensors["partials"] = graph.addVariable(FLOAT, {3}, "partials");
    graph.setTileMapping(tensors["partials"], 0);
    graph.setInitialValue(tensors["partials"], ArrayRef<float>{1, 2, 3});
    tensors["result"] = graph.addVariable(FLOAT, {10}, "result");
    graph.setTileMapping(tensors["result"], 0);
    tensors["idx"] = graph.addVariable(UNSIGNED_INT, {}, "idx");
    graph.setTileMapping(tensors["idx"], 0);
    tensors["1.0"] = graph.addConstant(FLOAT, {}, 1.0f, "1.0");
    graph.setTileMapping(tensors["1.0"], 0);
    tensors["1"] = graph.addConstant(UNSIGNED_INT, {}, 1, "1");
    graph.setTileMapping(tensors["1"], 0);

    graph.createHostRead("readResult", tensors["result"], false);


    auto cs = graph.addComputeSet("test");


    auto v = graph.addVertex(cs,
                             "AppendReducedSum",
                             {
                                     {"partials",    tensors["partials"].flatten()},
                                     {"index",       tensors["idx"]},
                                     {"numPartials", 3},
                                     {"finals",      tensors["result"].flatten()},
                             });
    auto reduceProg = Execute(cs);

    auto incrementProg = Sequence();
    popops::addInPlace(graph, tensors["idx"], tensors["1"], incrementProg, "idx++");
    popops::addInPlace(graph, tensors["partials"], tensors["1.0"], incrementProg, "partials++");

    auto finalProg = Repeat(10, Sequence{reduceProg, incrementProg});

    graph.setCycleEstimate(v, 1);
    graph.setTileMapping(v, 0);

    auto engine = lbm::createDebugEngine(graph, {finalProg});
    engine.load(device);
    engine.run();

    auto result = std::array<float, 10>{0};
    engine.readTensor("readResult", result.begin(), result.end());

    auto expected = std::array<float, 10>{
            1 + 2 + 3,
            2 + 3 + 4,
            3 + 4 + 5,
            4 + 5 + 6,
            5 + 6 + 7,
            6 + 7 + 8,
            7 + 8 + 9,
            8 + 9 + 10,
            9 + 10 + 11,
            10 + 11 + 12
    };
    ASSERT_EQ(result, expected);
}


TEST(averageVelocity, testFullAverage) {
    auto device = lbm::getIpuModel();
    auto graph = Graph{device.value().getTarget()};
    graph.addCodelets("D2Q9Codelets.cpp");
    popops::addCodelets(graph);

    auto tensors = lbm::TensorMap{};
    tensors["cells"] = graph.addVariable(FLOAT, {2, 2, 9}, "cells");
    graph.setTileMapping(tensors["cells"][0], 0);
    graph.setTileMapping(tensors["cells"][1], 1);
    graph.setInitialValue(tensors["cells"], ArrayRef<float>{1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                            0, 1, 2, 3, 4, 5, 6, 7, 8,
                                                            2, 3, 4, 5, 6, 7, 8, 9, 10,
                                                            2, 3, 4, 5, 6, 7, 8, 9, 10});
    tensors["obstacles"] = graph.addVariable(BOOL, {2, 2}, "obstacles");
    graph.setTileMapping(tensors["obstacles"][0], 0);
    graph.setTileMapping(tensors["obstacles"][1], 0);
    graph.setInitialValue(tensors["obstacles"], ArrayRef<bool>{true, false, true, true});
    tensors["partialsPerWorker"] = graph.addVariable(FLOAT, {4}, "partialsPerWorker");
    graph.setTileMapping(tensors["partialsPerWorker"].slice({0, 2}, 0), 0);
    graph.setTileMapping(tensors["partialsPerWorker"].slice({2, 4}, 0), 1);
    tensors["partialsPerTile"] = graph.addVariable(FLOAT, {2}, "partialsPerTile");
    graph.setTileMapping(tensors["partialsPerTile"][0], 0);
    graph.setTileMapping(tensors["partialsPerTile"][1], 1);
    tensors["result"] = graph.addVariable(FLOAT, {10}, "result");
    graph.setTileMapping(tensors["result"], 0);
    tensors["idx"] = graph.addVariable(UNSIGNED_INT, {}, "idx");
    graph.setTileMapping(tensors["idx"], 0);
    tensors["1.0"] = graph.addConstant(FLOAT, {}, 1.0f, "1.0");
    graph.setTileMapping(tensors["1.0"], 0);
    tensors["1"] = graph.addConstant(UNSIGNED_INT, {}, 1, "1");
    graph.setTileMapping(tensors["1"], 0);


    graph.createHostRead("readResult", tensors["result"], false);


    auto maskedSumCs = graph.addComputeSet("test");


    for (size_t i = 0; i < 4; i++) {
        auto v = graph.addVertex(maskedSumCs,
                                 "MaskedSumPartial",
                                 {
                                         {"cells",     tensors["cells"][i / 2][i % 2]},
                                         {"obstacles", tensors["obstacles"].slice({i / 2, i % 2},
                                                                                  {i / 2 + 1, i % 2 + 1}).flatten()},
                                         {"numCells",  1},
                                         {"out",       tensors["partialsPerWorker"][i]},
                                 });

        graph.setCycleEstimate(v, 1);
        graph.setTileMapping(v, i / 2);
    }

    auto partialReduceCs = graph.addComputeSet("partialReduceCs");

    for (size_t i = 0; i < 2; i++) {

        auto v = graph.addVertex(partialReduceCs,
                                 "ReducePartials",
                                 {
                                         {"partials",    tensors["partialsPerWorker"].slice({i * 2, i * 2 + 2},
                                                                                            0).flatten()},
                                         {"numPartials", 2},
                                         {"out",         tensors["partialsPerTile"][i]},
                                 });


        graph.setCycleEstimate(v, 1);
        graph.setTileMapping(v, i);
    }

    auto appendReduceCs = graph.addComputeSet("appendReduceCs");

    auto v = graph.addVertex(appendReduceCs,
                             "AppendReducedSum",
                             {
                                     {"partials",    tensors["partialsPerTile"].flatten()},
                                     {"index",       tensors["idx"]},
                                     {"numPartials", 2},
                                     {"finals",      tensors["result"].flatten()},
                             });
    graph.setCycleEstimate(v, 1);
    graph.setTileMapping(v, 0);

    auto reduceProg = Sequence{Execute(maskedSumCs), Execute(partialReduceCs), Execute(appendReduceCs)};

    auto incrementProg = Sequence();
    popops::addInPlace(graph, tensors["idx"], tensors["1"], incrementProg, "idx++");
    popops::addInPlace(graph, tensors["cells"], tensors["1.0"], incrementProg, "cells++");

    auto finalProg = Repeat(10, Sequence{reduceProg, incrementProg});

    auto engine = lbm::createDebugEngine(graph, {finalProg});
    engine.load(device.value());
    engine.run();

    auto result = std::array<float, 10>{0};
    engine.readTensor("readResult", result.begin(), result.end());

    auto expected = std::array<float, 10>{
            153, 180, 207, 234, 261, 288, 315, 342, 369, 396
    };
    ASSERT_EQ(result, expected);
}



