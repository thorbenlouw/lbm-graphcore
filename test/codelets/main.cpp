#include "gtest/gtest.h"

#include "../../main/GraphcoreUtils.hpp"
#include <poplar/Device.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Engine.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>
#include <math.h>

using namespace poplar;
using namespace poplar::program;

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}

TEST(averageVelocity, testNormedVelocityVertex) {
    auto device = lbm::getIpuDevice().value();
    auto graph = Graph{device.getTarget()};
    graph.addCodelets("D2Q9Codelets.cpp");
    auto tensors = lbm::TensorMap{};
    tensors["cells"] = graph.addVariable(FLOAT, {1, 3, 9}, "cells");
    graph.setTileMapping(tensors["cells"], 0);
    graph.setInitialValue(tensors["cells"], ArrayRef<float>{1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                            0, 1, 2, 3, 4, 5, 6, 7, 8,
                                                            2, 3, 4, 5, 6, 7, 8, 9, 10});

    tensors["vels"] = graph.addVariable(FLOAT, {2}, "vels");
    graph.setTileMapping(tensors["vels"], 0);

    graph.createHostRead("readResult", tensors["vels"], false);


    auto cs = graph.addComputeSet("test");

    auto v = graph.addVertex(cs,
                             "NormedVelocityVertex",
                             {
                                     {"cells",    tensors["cells"].flatten()},
                                     {"numCells", 3},
                                     {"vels",     tensors["vels"]},
                             });

    graph.setCycleEstimate(v, 1);
    graph.setTileMapping(v, 0);

    auto engine = lbm::createDebugEngine(graph, {Execute(cs)});
    engine.load(device);
    engine.run();

    auto vels = std::array<float, 3>{0.0f, 0.0f, 0.0f};
    engine.readTensor("readResult", vels.data());

    auto velocity = [](std::array<float, 9> cell) -> float {
        auto local_density = cell[0] + cell[1] + cell[2] + cell[3] + cell[4] +
                             cell[5] + cell[6] + cell[7] + cell[8];
        auto ux = cell[1] + cell[5] + cell[8] - (cell[3] + cell[6] + cell[7]) / local_density;
        auto uy = cell[2] + cell[5] + cell[6] - (cell[4] + cell[7] + cell[8]) / local_density;
        return sqrtf(ux * ux + uy * uy);
    };

    ASSERT_EQ(vels[0], velocity(std::array<float, 9>{1, 2, 3, 4, 5, 6, 7, 8, 9}));
    ASSERT_EQ(vels[1], velocity(std::array<float, 9>{0, 1, 2, 3, 4, 5, 6, 7, 8}));
    ASSERT_EQ(vels[2], velocity(std::array<float, 9>{2, 3, 4, 5, 6, 7, 8, 9, 10}));

}

/**
 * The maskedSumPartial is the first step of working out the average velocity. We take only those cells
 * which are not obstacles, and
 */
TEST(averageVelocity, testMaskedSumPartial) {
    auto device = poplar::Device::createCPUDevice();
    auto graph = Graph{device.getTarget()};
    graph.addCodelets("D2Q9Codelets.cpp");
    auto tensors = lbm::TensorMap{};
    tensors["cells"] = graph.addVariable(FLOAT, {1, 3}, "cells");
    graph.setTileMapping(tensors["velocities"], 0);
    graph.setInitialValue(tensors["velocities"], ArrayRef<float>{1, 100, 1000});
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
                                     {"velocities",    tensors["velocities"].flatten()},
                                     {"obstacles",     tensors["obstacles"].flatten()},
                                     {"numCells",      3},
                                     {"totalAndCount", tensors["result"]},
                             });

    graph.setCycleEstimate(v, 1);
    graph.setTileMapping(v, 0);

    auto engine = lbm::createDebugEngine(graph, {Execute(cs)});
    engine.load(device);
    engine.run();

    float result[2] = {0.0};
    engine.readTensor("readResult", &result);

    ASSERT_EQ(result[0], 1 + 1000); // the two non-obstacled cells added
    ASSERT_EQ(result[1], 2); // the count of non-obstacled cells


}


TEST(averageVelocity, testReducePartials) {
    auto device = poplar::Device::createCPUDevice();
    auto graph = Graph{device.getTarget()};
    graph.addCodelets("D2Q9Codelets.cpp");
    auto tensors = lbm::TensorMap{};
    tensors["partials"] = graph.addVariable(FLOAT, {3 * 2}, "partials");
    graph.setTileMapping(tensors["partials"], 0);
    graph.setInitialValue(tensors["partials"], ArrayRef<float>{100, 2, 300, 3, 1, 1});
    tensors["result"] = graph.addVariable(FLOAT, {}, "result");
    graph.setTileMapping(tensors["result"], 0);


    graph.createHostRead("readResult", tensors["result"], false);


    auto cs = graph.addComputeSet("test");

    auto v = graph.addVertex(cs,
                             "ReducePartials",
                             {
                                     {"totalAndCountPartials", tensors["partials"].flatten()},
                                     {"numPartials",           3},
                                     {"totalAndCount",         tensors["result"]},
                             });

    graph.setCycleEstimate(v, 1);
    graph.setTileMapping(v, 0);

    auto engine = lbm::createDebugEngine(graph, {Execute(cs)});
    engine.load(device);
    engine.run();

    float result[2] = {0, 0};
    engine.readTensor("readResult", &result);

    ASSERT_EQ(result[0], 100 + 300 + 1);
    ASSERT_EQ(result[1], 2 + 3 + 1);


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


TEST(accelerate, testAccelerateVertex) {
    auto device = poplar::Device::createCPUDevice();
    auto graph = Graph{device.getTarget()};
    graph.addCodelets("D2Q9Codelets.cpp");
    auto tensors = lbm::TensorMap{};

    auto nx = 3u;
    auto ny = 2u;
    auto accel = 1u;
    auto density = 9u;

    tensors["cells"] = graph.addVariable(FLOAT, {ny, nx, 9}, "cells");
    graph.setTileMapping(tensors["cells"], 0);
    graph.setInitialValue(tensors["cells"],
                          ArrayRef<float>{1, 0.5, 1, 1, 1, 1, 1, 1, 1,
                                          0, 1, 2, 3, 4, 5, 6, 7, 8,
                                          0, 1, 2, 3, 4, 5, 6, 7, 8,
                                          2, 3, 4, 5, 6, 7, 8, 9, 10,
                                          2, 3, 4, 5, 6, 7, 8, 9, 10,
                                          2, 3, 4, 5, 6, 7, 8, 9, 10});
    tensors["obstacles"] = graph.addVariable(BOOL, {ny, nx}, "obstacles");
    graph.setTileMapping(tensors["obstacles"], 0);
    graph.setInitialValue(tensors["obstacles"],
                          ArrayRef<bool>{
                                  false, true, false,
                                  false, true, true});

    graph.createHostRead("readCells", tensors["cells"], false);


    auto cs = graph.addComputeSet("test");

    auto v = graph.addVertex(cs,
                             "AccelerateFlowVertex",
                             {
                                     {"cellsInSecondRow",     tensors["cells"][ny - 2].flatten()},
                                     {"obstaclesInSecondRow", tensors["obstacles"][ny - 2].flatten()},
                                     {"density",              density},
                                     {"partitionWidth",       nx},
                                     {"accel",                accel},
                             });

    graph.setCycleEstimate(v, 1);
    graph.setTileMapping(v, 0);

    auto prog = Sequence(Execute(cs)
//            PrintTensor(tensors["cells"])
    );
    auto engine = lbm::createDebugEngine(graph, {prog});
    engine.load(device);
    engine.run();

    auto cells = std::array<std::array<std::array<float, 9>, 3>, 2>();
    engine.readTensor("readCells", &cells);

    const auto w1 = 1.0f;
    const auto w2 = 0.25f;
    // w1 = 1, w2 = 0.25 Should not update when when cell is obstacle everything in row 2!)
    auto expected = std::array<std::array<std::array<float, 9>, 3>, 2>();
    expected[0][0] = {1, 0.5, 1, 1, 1, 1, 1, 1, 1}; // doesn't change because of negative conditions
    expected[0][1] = {0, 1, 2, 3, 4, 5, 6, 7, 8}; // doesn't change because obstacle
    expected[0][2] = {0, 1 + w1, 2, 3 - w1, 4, 5 + w2,
                      6 - w2, 7 - w2,
                      8 + w2}; // does change
    expected[1][0] = {2, 3, 4, 5, 6, 7, 8, 9, 10}; // these don't change (only 2nd row from bottom)
    expected[1][1] = {2, 3, 4, 5, 6, 7, 8, 9, 10};  // these don't change (only 2nd row from bottom)
    expected[1][2] = {2, 3, 4, 5, 6, 7, 8, 9, 10};// these don't change (only 2nd row from bottom)

    ASSERT_EQ(cells[0][0], expected[0][0]) << "cells[0,0,:] doesn't change because of negative conditions";
    ASSERT_EQ(cells[0][1], expected[0][1]) << "cells[0,1,:] doesn't change because obstacle";
    ASSERT_EQ(cells[0][2], expected[0][2]) << "cells[0,2,:] does change";
    ASSERT_EQ(cells[1][0], expected[1][0]) << "cells[1,0,:] no change (only 2nd row from bottom)";
    ASSERT_EQ(cells[1][1], expected[1][1]) << "cells[1,1,:] no change (only 2nd row from bottom)";
    ASSERT_EQ(cells[1][2], expected[1][2]) << "cells[1,2,:] no change (only 2nd row from bottom)";


}


TEST(propagate, testPropagateVertex) {
    auto device = poplar::Device::createCPUDevice();
    auto graph = Graph{device.getTarget()};
    graph.addCodelets("D2Q9Codelets.cpp");
    auto tensors = lbm::TensorMap{};

    auto nx = 3u;
    auto ny = 2u;
    auto accel = 1u;
    auto density = 9u;

    tensors["cells"] = graph.addVariable(FLOAT, {ny, nx, 9}, "cells");
    graph.setTileMapping(tensors["cells"], 0);
    graph.setInitialValue(tensors["cells"],
                          ArrayRef<float>{1, 0.5, 1, 1, 1, 1, 1, 1, 1,
                                          0, 1, 2, 3, 4, 5, 6, 7, 8,
                                          0, 1, 2, 3, 4, 5, 6, 7, 8,
                                          2, 3, 4, 5, 6, 7, 8, 9, 10,
                                          2, 3, 4, 5, 6, 7, 8, 9, 10,
                                          2, 3, 4, 5, 6, 7, 8, 9, 10});
    tensors["obstacles"] = graph.addVariable(BOOL, {ny, nx}, "obstacles");
    graph.setTileMapping(tensors["obstacles"], 0);
    graph.setInitialValue(tensors["obstacles"],
                          ArrayRef<bool>{
                                  false, true, false,
                                  false, true, true});

    graph.createHostRead("readCells", tensors["cells"], false);


    auto cs = graph.addComputeSet("test");

    auto v = graph.addVertex(cs,
                             "AccelerateFlowVertex",
                             {
                                     {"cellsInSecondRow",     tensors["cells"][ny - 2].flatten()},
                                     {"obstaclesInSecondRow", tensors["obstacles"][ny - 2].flatten()},
                                     {"density",              density},
                                     {"partitionWidth",       nx},
                                     {"accel",                accel},
                             });

    graph.setCycleEstimate(v, 1);
    graph.setTileMapping(v, 0);

    auto prog = Sequence(Execute(cs)
//            PrintTensor(tensors["cells"])
    );
    auto engine = lbm::createDebugEngine(graph, {prog});
    engine.load(device);
    engine.run();

    auto cells = std::array<std::array<std::array<float, 9>, 3>, 2>();
    engine.readTensor("readCells", &cells);

    const auto w1 = 1.0f;
    const auto w2 = 0.25f;
    // w1 = 1, w2 = 0.25 Should not update when when cell is obstacle everything in row 2!)
    auto expected = std::array<std::array<std::array<float, 9>, 3>, 2>();
    expected[0][0] = {1, 0.5, 1, 1, 1, 1, 1, 1, 1}; // doesn't change because of negative conditions
    expected[0][1] = {0, 1, 2, 3, 4, 5, 6, 7, 8}; // doesn't change because obstacle
    expected[0][2] = {0, 1 + w1, 2, 3 - w1, 4, 5 + w2,
                      6 - w2, 7 - w2,
                      8 + w2}; // does change
    expected[1][0] = {2, 3, 4, 5, 6, 7, 8, 9, 10}; // these don't change (only 2nd row from bottom)
    expected[1][1] = {2, 3, 4, 5, 6, 7, 8, 9, 10};  // these don't change (only 2nd row from bottom)
    expected[1][2] = {2, 3, 4, 5, 6, 7, 8, 9, 10};// these don't change (only 2nd row from bottom)

    ASSERT_EQ(cells[0][0], expected[0][0]) << "cells[0,0,:] doesn't change because of negative conditions";
    ASSERT_EQ(cells[0][1], expected[0][1]) << "cells[0,1,:] doesn't change because obstacle";
    ASSERT_EQ(cells[0][2], expected[0][2]) << "cells[0,2,:] does change";
    ASSERT_EQ(cells[1][0], expected[1][0]) << "cells[1,0,:] no change (only 2nd row from bottom)";
    ASSERT_EQ(cells[1][1], expected[1][1]) << "cells[1,1,:] no change (only 2nd row from bottom)";
    ASSERT_EQ(cells[1][2], expected[1][2]) << "cells[1,2,:] no change (only 2nd row from bottom)";


}

