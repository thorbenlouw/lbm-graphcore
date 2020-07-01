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
    auto device = lbm::getIpuModel().value();
    auto graph = Graph{device.getTarget()};
    graph.addCodelets("D2Q9Codelets.cpp");
    auto tensors = lbm::TensorMap{};
    tensors["cells"] = graph.addVariable(FLOAT, {1, 3, 9}, "cells");
    graph.setTileMapping(tensors["cells"], 0);
    graph.setInitialValue(tensors["cells"], ArrayRef<float>{1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                            0, 1, 2, 3, 4, 5, 6, 7, 8,
                                                            2, 3, 4, 5, 6, 7, 8, 9, 10});

    tensors["vels"] = graph.addVariable(FLOAT, {3}, "vels");
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
        auto ux = (cell[1] + cell[5] + cell[8] - (cell[3] + cell[6] + cell[7])) / local_density;
        auto uy = (cell[2] + cell[5] + cell[6] - (cell[4] + cell[7] + cell[8])) / local_density;
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
    tensors["velocities"] = graph.addVariable(FLOAT, {1, 3}, "cells");
    graph.setTileMapping(tensors["velocities"], 0);
    graph.setInitialValue(tensors["velocities"], ArrayRef<float>{1, 100, 1000});
    tensors["obstacles"] = graph.addVariable(BOOL, {3}, "obstacles");
    graph.setTileMapping(tensors["obstacles"], 0);
    graph.setInitialValue(tensors["obstacles"], ArrayRef<bool>{false, true, false});

    tensors["result"] = graph.addVariable(FLOAT, {2}, "result");
    graph.setTileMapping(tensors["result"], 0);

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

    ASSERT_FLOAT_EQ(result[0], 1 + 1000); // the two non-obstacled cells added
    ASSERT_FLOAT_EQ(result[1], 2); // the count of non-obstacled cells


}


TEST(averageVelocity, testReducePartials) {
    auto device = poplar::Device::createCPUDevice();
    auto graph = Graph{device.getTarget()};
    graph.addCodelets("D2Q9Codelets.cpp");
    auto tensors = lbm::TensorMap{};
    tensors["partials"] = graph.addVariable(FLOAT, {3 * 2}, "partials");
    graph.setTileMapping(tensors["partials"], 0);
    graph.setInitialValue(tensors["partials"], ArrayRef<float>{100, 2, 300, 3, 1, 1});
    tensors["result"] = graph.addVariable(FLOAT, {2}, "result");
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

    ASSERT_FLOAT_EQ(result[0], 100 + 300 + 1);
    ASSERT_FLOAT_EQ(result[1], 2 + 3 + 1);


}


TEST(averageVelocity, testAppendReducedSum) {
    auto device = poplar::Device::createCPUDevice();
    auto graph = Graph{device.getTarget()};
    graph.addCodelets("D2Q9Codelets.cpp");
    popops::addCodelets(graph);

    auto tensors = lbm::TensorMap{};
    tensors["partials"] = graph.addVariable(FLOAT, {3 * 2}, "partials");
    graph.setTileMapping(tensors["partials"], 0);
    graph.setInitialValue(tensors["partials"], ArrayRef<float>{10, 5, 2, 1, 3, 9});
    tensors["result"] = graph.addVariable(FLOAT, {3}, "result");
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
                                     {"totalAndCountPartials", tensors["partials"].flatten()},
                                     {"index",                 tensors["idx"]},
                                     {"numPartials",           3},
                                     {"finals",                tensors["result"].flatten()},
                             });
    auto reduceProg = Execute(cs);

    auto incrementProg = Sequence();
    popops::addInPlace(graph, tensors["idx"], tensors["1"], incrementProg, "idx++");
    popops::addInPlace(graph, tensors["partials"], tensors["1.0"], incrementProg, "partials++");

    auto finalProg = Repeat(3, Sequence{reduceProg, incrementProg});

    graph.setCycleEstimate(v, 1);
    graph.setTileMapping(v, 0);

    auto engine = lbm::createDebugEngine(graph, {finalProg});
    engine.load(device);
    engine.run();

    auto result = std::array<float, 3>{0};
    engine.readTensor("readResult", result.begin(), result.end());

    auto expected = std::array<float, 3>{
            (10.0 + 2 + 3) / (5 + 1 + 9),
            (11.0 + 3 + 4) / (6 + 2 + 10),
            (12.0 + 4 + 5) / (7 + 3 + 11),

    };
    ASSERT_FLOAT_EQ(result[0], expected[0]);
    ASSERT_FLOAT_EQ(result[1], expected[1]);
    ASSERT_FLOAT_EQ(result[2], expected[2]);
};

/**
 * A full end-to-end test of the average velocity operation, in which we
 * (1) Take an input tensor of cells that is 2x2(x9) mapped onto 2 workers each of 2 tiles
 * (2) Calculate the normed velocities
 * (3) Use the obstacles mask and reduce to the partial representing the non-obstacled averages in the worker -> (sum, count)
 * (4) Reduce the workers' answers per tile -> (sum, count)
 * (5) Reduce all the tiles' answers and store the result at the end of the results vector
 *  Then we increment all the cell values (to have different test values) and repeat so that we have a results vector of 3
 *
 *  This is similar to how we will track the average velocity at each iteration of the simulation
 */
TEST(averageVelocity, testFullAverage) {
    auto device = lbm::getIpuModel();
    auto graph = Graph{device.value().getTarget()};
    graph.addCodelets("D2Q9Codelets.cpp");
    popops::addCodelets(graph);

    const auto cells = std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9,
                                          0, 1, 2, 3, 4, 5, 6, 7, 8,
                                          2, 3, 4, 5, 6, 7, 8, 9, 10,
                                          2, 3, 4, 5, 6, 7, 8, 9, 10};

    auto tensors = lbm::TensorMap{};
    tensors["cells"] = graph.addVariable(FLOAT, {2, 2, 9}, "cells");
    graph.setTileMapping(tensors["cells"][0], 0);
    graph.setTileMapping(tensors["cells"][1], 1);
    graph.setInitialValue(tensors["cells"], ArrayRef<float>{1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                            0, 1, 2, 3, 4, 5, 6, 7, 8,
                                                            2, 3, 4, 5, 6, 7, 8, 9, 10,
                                                            2, 3, 4, 5, 6, 7, 8, 9, 10});
    tensors["velocities"] = graph.addVariable(FLOAT, {2, 2}, "velocities");
    graph.setTileMapping(tensors["velocities"][0], 0);
    graph.setTileMapping(tensors["velocities"][1], 1);

    tensors["obstacles"] = graph.addVariable(BOOL, {2, 2}, "obstacles");
    graph.setTileMapping(tensors["obstacles"][0], 0);
    graph.setTileMapping(tensors["obstacles"][1], 0);
    graph.setInitialValue(tensors["obstacles"], ArrayRef<bool>{true, false, false, true});
    tensors["partialsPerWorker"] = graph.addVariable(FLOAT, {4 * 2}, "partialsPerWorker");
    graph.setTileMapping(tensors["partialsPerWorker"].slice({0, 4}, 0), 0);
    graph.setTileMapping(tensors["partialsPerWorker"].slice({4, 8}, 0), 1);
    tensors["partialsPerTile"] = graph.addVariable(FLOAT, {2 * 2}, "partialsPerTile");
    graph.setTileMapping(tensors["partialsPerTile"].slice(0, 2, 0), 0);
    graph.setTileMapping(tensors["partialsPerTile"].slice(2, 4, 0), 1);
    tensors["result"] = graph.addVariable(FLOAT, {3}, "result");
    graph.setTileMapping(tensors["result"], 0);
    tensors["idx"] = graph.addVariable(UNSIGNED_INT, {}, "idx");
    graph.setTileMapping(tensors["idx"], 0);
    tensors["1.0"] = graph.addConstant(FLOAT, {}, 1.0f, "1.0");
    graph.setTileMapping(tensors["1.0"], 0);
    tensors["1"] = graph.addConstant(UNSIGNED_INT, {}, 1, "1");
    graph.setTileMapping(tensors["1"], 0);


    graph.createHostRead("readResult", tensors["result"], false);


    auto velocitiesCs = graph.addComputeSet("velocitiesCs");

    for (size_t i = 0; i < 4; i++) {

        auto v = graph.addVertex(velocitiesCs,
                                 "NormedVelocityVertex",
                                 {
                                         {"cells",    tensors["cells"][i / 2][i % 2]},
                                         {"numCells", 1},
                                         {"vels",     tensors["velocities"].slice({i / 2, i % 2},
                                                                                  {i / 2 + 1, i % 2 + 1}).flatten()},
                                 });

        graph.setCycleEstimate(v, 1);
        graph.setTileMapping(v, i / 2);
    }
    auto maskedSumCs = graph.addComputeSet("test");

    for (size_t i = 0; i < 4; i++) {
        auto v = graph.addVertex(maskedSumCs,
                                 "MaskedSumPartial",
                                 {
                                         {"velocities",    tensors["velocities"].slice({i / 2, i % 2},
                                                                                       {i / 2 + 1,
                                                                                        i % 2 + 1}).flatten()},
                                         {"obstacles",     tensors["obstacles"].slice({i / 2, i % 2},
                                                                                      {i / 2 + 1,
                                                                                       i % 2 + 1}).flatten()},
                                         {"numCells",      1},
                                         {"totalAndCount", tensors["partialsPerWorker"].slice(i * 2, i * 2 + 1,
                                                                                              0).flatten()},
                                 });

        graph.setCycleEstimate(v, 1);
        graph.setTileMapping(v, i / 2);
    }

    auto partialReduceCs = graph.addComputeSet("partialReduceCs");

    for (size_t i = 0; i < 2; i++) {

        auto v = graph.addVertex(partialReduceCs,
                                 "ReducePartials",
                                 {
                                         {"totalAndCountPartials", tensors["partialsPerWorker"].slice(
                                                 {i * 4, i * 4 + 4},
                                                 0).flatten()},
                                         {"numPartials",           2},
                                         {"totalAndCount",         tensors["partialsPerTile"].slice(i * 2, i * 2 + 1,
                                                                                                    0).flatten()},
                                 });


        graph.setCycleEstimate(v, 1);
        graph.setTileMapping(v, i);
    }

    auto appendReduceCs = graph.addComputeSet("appendReduceCs");

    auto v = graph.addVertex(appendReduceCs,
                             "AppendReducedSum",
                             {
                                     {"totalAndCountPartials", tensors["partialsPerTile"].flatten()},
                                     {"index",                 tensors["idx"]},
                                     {"numPartials",           2},
                                     {"finals",                tensors["result"].flatten()},
                             });
    graph.setCycleEstimate(v, 1);
    graph.setTileMapping(v, 0);

    auto reduceProg = Sequence{Execute(velocitiesCs), Execute(maskedSumCs), Execute(partialReduceCs),
                               Execute(appendReduceCs)};

    auto incrementProg = Sequence();
    popops::addInPlace(graph, tensors["idx"], tensors["1"], incrementProg, "idx++");
    popops::addInPlace(graph, tensors["cells"], tensors["1.0"], incrementProg, "cells++");

    auto finalProg = Repeat(3, Sequence{reduceProg, incrementProg,
//                                        PrintTensor(tensors["cells"]),
//                                        PrintTensor(tensors["velocities"]),
//                                        PrintTensor(tensors["partialsPerWorker"]),
//                                        PrintTensor(tensors["partialsPerTile"]),
//                                        PrintTensor(tensors["result"]),
    });

    auto engine = lbm::createDebugEngine(graph, {finalProg});
    engine.load(device.value());
    engine.run();

    auto result = std::array<float, 3>{0};
    engine.readTensor("readResult", result.begin(), result.end());


    auto velocityFn = [](const int offset, const std::vector<float> &cell) -> float {
        auto bump = offset * 9;
        auto local_density = cell[bump + 0] + cell[bump + 1] + cell[bump + 2] + cell[bump + 3] + cell[bump + 4] +
                             cell[bump + 5] + cell[bump + 6] + cell[bump + 7] + cell[bump + 8];
        auto ux = (cell[bump + 1] + cell[bump + 5] + cell[bump + 8] -
                   (cell[bump + 3] + cell[bump + 6] + cell[bump + 7])) / local_density;
        auto uy = (cell[bump + 2] + cell[bump + 5] + cell[bump + 6] -
                   (cell[bump + 4] + cell[bump + 7] + cell[bump + 8])) / local_density;
        return sqrtf(ux * ux + uy * uy);
    };

    auto averageVelocityFn = [&velocityFn](
            const std::vector<float> &cells) -> float { // Remember only the middle 2 are not obstacles
        return (velocityFn(0, cells) * 0 +
                velocityFn(1, cells) * 1 +
                velocityFn(2, cells) * 1 +
                velocityFn(3, cells) * 0) / 2;
    };

    auto cells1 = std::vector<float>(4 * 9);
    auto cells2 = std::vector<float>(4 * 9);

    auto incrementFn = [](float a) -> float { return a + 1; };
    std::transform(cells.begin(), cells.end(), cells1.begin(), incrementFn);
    std::transform(cells.begin(), cells.end(), cells2.begin(), incrementFn);

    ASSERT_NEAR(result[0], averageVelocityFn(cells), 1E-5);
    ASSERT_NEAR(result[1], averageVelocityFn(cells1), 1E-5);
    ASSERT_NEAR(result[2], averageVelocityFn(cells2), 1E-5);
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

enum SpeedIndexes {
    Middle, East, North, West, South, NorthEast, NorthWest, SouthWest, SouthEast
};


void createCellsAndHalos(lbm::TensorMap &tensors, Graph &graph) {
    auto nx = 5u; // excluding halo
    auto ny = 3u;
    /* Input test data:
     * (each of the 9 speeds will be +0.01*idx, so offset 3 is +0.03)
     *                                      (6,4) = top right
     *  -----------------------------------
     *  | 1.0 | 2.0 3.0 4.0 5.0 6.0 | 7.0 |
     *  -----------------------------------
     *  | 1.1 | 2.1 3.1 4.1 5.1 6.1 | 7.1 |
     *  | 1.2 | 2.2 3.2 4.2 5.2 6.2 | 7.2 |
     *  | 1.3 | 2.3 3.3 4.3 5.3 6.3 | 7.3 |
     *  -----------------------------------
     *  | 1.4 | 2.4 3.4 4.4 5.4 6.4 | 7.4 |
     *  -----------------------------------
     *  (0,0)=bottom left
     */
    tensors["cells"] = graph.addVariable(FLOAT, {ny, nx, 9}, "cells");
    graph.setTileMapping(tensors["cells"], 0);
    graph.setInitialValue(tensors["cells"],
                          ArrayRef<float>{
                                  2.30, 2.31, 2.32, 2.33, 2.34, 2.35, 2.36, 2.37, 2.38,
                                  3.30, 3.31, 3.32, 3.33, 3.34, 3.35, 3.36, 3.37, 3.38,
                                  4.30, 4.31, 4.32, 4.33, 4.34, 4.35, 4.36, 4.37, 4.38,
                                  5.30, 5.31, 5.32, 5.33, 5.34, 5.35, 5.36, 5.37, 5.38,
                                  6.30, 6.31, 6.32, 6.33, 6.34, 6.35, 6.36, 6.37, 6.38,

                                  2.20, 2.21, 2.22, 2.23, 2.24, 2.25, 2.26, 2.27, 2.28,
                                  3.20, 3.21, 3.22, 3.23, 3.24, 3.25, 3.26, 3.27, 3.28,
                                  4.20, 4.21, 4.22, 4.23, 4.24, 4.25, 4.26, 4.27, 4.28,
                                  5.20, 5.21, 5.22, 5.23, 5.24, 5.25, 5.26, 5.27, 5.28,
                                  6.20, 6.21, 6.22, 6.23, 6.24, 6.25, 6.26, 6.27, 6.28,

                                  2.10, 2.11, 2.12, 2.13, 2.14, 2.15, 2.16, 2.17, 2.18,
                                  3.10, 3.11, 3.12, 3.13, 3.14, 3.15, 3.16, 3.17, 3.18,
                                  4.10, 4.11, 4.12, 4.13, 4.14, 4.15, 4.16, 4.17, 4.18,
                                  5.10, 5.11, 5.12, 5.13, 5.14, 5.15, 5.16, 5.17, 5.18,
                                  6.10, 6.11, 6.12, 6.13, 6.14, 6.15, 6.16, 6.17, 6.18,
                          });

    tensors["tmp_cells"] = graph.addVariable(FLOAT, {ny, nx, 9}, "tmp_cells");
    graph.setTileMapping(tensors["tmp_cells"], 0);


    tensors["haloTop"] = graph.addVariable(FLOAT, {nx, 9}, "haloTop");
    graph.setTileMapping(tensors["haloTop"], 0);
    graph.setInitialValue(tensors["haloTop"],
                          ArrayRef<float>{
                                  2.00, 2.01, 2.02, 2.03, 2.04, 2.05, 2.06, 2.07, 2.08,
                                  3.00, 3.01, 3.02, 3.03, 3.04, 3.05, 3.06, 3.07, 3.08,
                                  4.00, 4.01, 4.02, 4.03, 4.04, 4.05, 4.06, 4.07, 4.08,
                                  5.00, 5.01, 5.02, 5.03, 5.04, 5.05, 5.06, 5.07, 5.08,
                                  6.00, 6.01, 6.02, 6.03, 6.04, 6.05, 6.06, 6.07, 6.08,
                          });

    tensors["haloBottom"] = graph.addVariable(FLOAT, {nx, 9}, "haloBottom");
    graph.setTileMapping(tensors["haloBottom"], 0);
    graph.setInitialValue(tensors["haloBottom"],
                          ArrayRef<float>{
                                  2.40, 2.41, 2.42, 2.43, 2.44, 2.45, 2.46, 2.47, 2.48,
                                  3.40, 3.41, 3.42, 3.43, 3.44, 3.45, 3.46, 3.47, 3.48,
                                  4.40, 4.41, 4.42, 4.43, 4.44, 4.45, 4.46, 4.47, 4.48,
                                  5.40, 5.41, 5.42, 5.43, 5.44, 5.45, 5.46, 5.47, 5.48,
                                  6.40, 6.41, 6.42, 6.43, 6.44, 6.45, 6.46, 6.47, 6.48,
                          });

    tensors["haloLeft"] = graph.addVariable(FLOAT, {ny, 9}, "haloLeft");
    graph.setTileMapping(tensors["haloLeft"], 0);
    graph.setInitialValue(tensors["haloLeft"],
                          ArrayRef<float>{
                                  1.30, 1.31, 1.32, 1.33, 1.34, 1.35, 1.36, 1.37, 1.38,
                                  1.20, 1.21, 1.22, 1.23, 1.24, 1.25, 1.26, 1.27, 1.28,
                                  1.10, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18,
                          });

    tensors["haloRight"] = graph.addVariable(FLOAT, {ny, 9}, "haloRight");
    graph.setTileMapping(tensors["haloRight"], 0);
    graph.setInitialValue(tensors["haloRight"],
                          ArrayRef<float>{
                                  7.30, 7.31, 7.32, 7.33, 7.34, 7.35, 7.36, 7.37, 7.38,
                                  7.20, 7.21, 7.22, 7.23, 7.24, 7.25, 7.26, 7.27, 7.28,
                                  7.10, 7.11, 7.12, 7.13, 7.14, 7.15, 7.16, 7.17, 7.18,
                          });

    tensors["haloTopLeft"] = graph.addVariable(FLOAT, {}, "haloTopLeft");
    graph.setTileMapping(tensors["haloTopLeft"], 0);
    graph.setInitialValue(tensors["haloTopLeft"], 1.0 + 0.01 * SpeedIndexes::NorthWest);

    tensors["haloTopRight"] = graph.addVariable(FLOAT, {}, "haloTopRight");
    graph.setTileMapping(tensors["haloTopRight"], 0);
    graph.setInitialValue(tensors["haloTopRight"], 7.0 + 0.01 * SpeedIndexes::NorthEast);

    tensors["haloBottomRight"] = graph.addVariable(FLOAT, {}, "haloBottomRight");
    graph.setTileMapping(tensors["haloBottomRight"], 0);
    graph.setInitialValue(tensors["haloBottomRight"], 7.4 + 0.01 * SpeedIndexes::SouthEast);


    tensors["haloBottomLeft"] = graph.addVariable(FLOAT, {}, "haloBottomLeft");
    graph.setTileMapping(tensors["haloBottomLeft"], 0);
    graph.setInitialValue(tensors["haloBottomLeft"], 1.4 + 0.01 * SpeedIndexes::SouthWest);
}


std::array<std::array<std::array<float, 9>, 5>, 3> runPropagate() {
    auto device = poplar::Device::createCPUDevice();
    auto graph = Graph{device.getTarget()};
    graph.addCodelets("D2Q9Codelets.cpp");
    auto tensors = lbm::TensorMap{};

    const auto nx = 5u; // excluding halo
    const auto ny = 3u;

    createCellsAndHalos(tensors, graph);

    graph.createHostRead("readTmpCells", tensors["tmp_cells"], false);

    auto cs = graph.addComputeSet("test");

    auto v = graph.addVertex(cs,
                             "PropagateVertex",
                             {
                                     {"in",              tensors["cells"].flatten()},
                                     {"out",             tensors["tmp_cells"].flatten()},
                                     {"numRows",         ny},
                                     {"numCols",         nx},
                                     {"haloTop",         tensors["haloTop"].flatten()},
                                     {"haloBottom",      tensors["haloBottom"].flatten()},
                                     {"haloLeft",        tensors["haloLeft"].flatten()},
                                     {"haloRight",       tensors["haloRight"].flatten()},
                                     {"haloTopLeft",     tensors["haloTopLeft"]},
                                     {"haloTopRight",    tensors["haloTopRight"]},
                                     {"haloBottomLeft",  tensors["haloBottomLeft"]},
                                     {"haloBottomRight", tensors["haloBottomRight"]},
                             });

    graph.setCycleEstimate(v, 1);
    graph.setTileMapping(v, 0);

    auto prog = Sequence(Execute(cs)
//                        ,PrintTensor(tensors["cells"]),
//                         PrintTensor(tensors["tmp_cells"]

    );
    auto engine = lbm::createDebugEngine(graph, {prog});
    engine.load(device);
    engine.run();

    auto tmp_cells = std::array<std::array<std::array<float, 9>, nx>, ny>();
    engine.readTensor("readTmpCells", &tmp_cells);
    return std::move(tmp_cells);
}


TEST(propagate, testPropagateVertexTopLeft) {
    const auto nx = 5u; // excluding halo
    const auto ny = 3u;
    auto tmp_cells = runPropagate();

    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][0][SpeedIndexes::NorthWest], 1.0 + 0.01 * SpeedIndexes::NorthWest);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][0][SpeedIndexes::North], 2.0 + 0.01 * SpeedIndexes::North);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][0][SpeedIndexes::NorthEast], 3.0 + 0.01 * SpeedIndexes::NorthEast);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][0][SpeedIndexes::West], 1.1 + 0.01 * SpeedIndexes::West);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][0][SpeedIndexes::Middle], 2.1 + 0.01 * SpeedIndexes::Middle);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][0][SpeedIndexes::East], 3.1 + 0.01 * SpeedIndexes::East);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][0][SpeedIndexes::SouthWest], 1.2 + 0.01 * SpeedIndexes::SouthWest);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][0][SpeedIndexes::South], 2.2 + 0.01 * SpeedIndexes::South);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][0][SpeedIndexes::SouthEast], 3.2 + 0.01 * SpeedIndexes::SouthEast);
}


TEST(propagate, testPropagateVertexTop) {
    const auto nx = 5u; // excluding halo
    const auto ny = 3u;
    auto tmp_cells = runPropagate();

    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][1][SpeedIndexes::NorthWest], 2.0 + 0.01 * SpeedIndexes::NorthWest);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][1][SpeedIndexes::North], 3.0 + 0.01 * SpeedIndexes::North);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][1][SpeedIndexes::NorthEast], 4.0 + 0.01 * SpeedIndexes::NorthEast);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][1][SpeedIndexes::West], 2.1 + 0.01 * SpeedIndexes::West);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][1][SpeedIndexes::Middle], 3.1 + 0.01 * SpeedIndexes::Middle);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][1][SpeedIndexes::East], 4.1 + 0.01 * SpeedIndexes::East);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][1][SpeedIndexes::SouthWest], 2.2 + 0.01 * SpeedIndexes::SouthWest);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][1][SpeedIndexes::South], 3.2 + 0.01 * SpeedIndexes::South);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][1][SpeedIndexes::SouthEast], 4.2 + 0.01 * SpeedIndexes::SouthEast);

    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][2][SpeedIndexes::NorthWest], 3.0 + 0.01 * SpeedIndexes::NorthWest);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][2][SpeedIndexes::North], 4.0 + 0.01 * SpeedIndexes::North);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][2][SpeedIndexes::NorthEast], 5.0 + 0.01 * SpeedIndexes::NorthEast);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][2][SpeedIndexes::West], 3.1 + 0.01 * SpeedIndexes::West);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][2][SpeedIndexes::Middle], 4.1 + 0.01 * SpeedIndexes::Middle);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][2][SpeedIndexes::East], 5.1 + 0.01 * SpeedIndexes::East);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][2][SpeedIndexes::SouthWest], 3.2 + 0.01 * SpeedIndexes::SouthWest);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][2][SpeedIndexes::South], 4.2 + 0.01 * SpeedIndexes::South);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][2][SpeedIndexes::SouthEast], 5.2 + 0.01 * SpeedIndexes::SouthEast);

    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][3][SpeedIndexes::NorthWest], 4.0 + 0.01 * SpeedIndexes::NorthWest);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][3][SpeedIndexes::North], 5.0 + 0.01 * SpeedIndexes::North);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][3][SpeedIndexes::NorthEast], 6.0 + 0.01 * SpeedIndexes::NorthEast);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][3][SpeedIndexes::West], 4.1 + 0.01 * SpeedIndexes::West);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][3][SpeedIndexes::Middle], 5.1 + 0.01 * SpeedIndexes::Middle);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][3][SpeedIndexes::East], 6.1 + 0.01 * SpeedIndexes::East);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][3][SpeedIndexes::SouthWest], 4.2 + 0.01 * SpeedIndexes::SouthWest);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][3][SpeedIndexes::South], 5.2 + 0.01 * SpeedIndexes::South);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][3][SpeedIndexes::SouthEast], 6.2 + 0.01 * SpeedIndexes::SouthEast);
}

TEST(propagate, testPropagateVertexBottom) {
    auto tmp_cells = runPropagate();

    ASSERT_FLOAT_EQ(tmp_cells[0][1][SpeedIndexes::NorthWest], 2.2 + 0.01 * SpeedIndexes::NorthWest);
    ASSERT_FLOAT_EQ(tmp_cells[0][1][SpeedIndexes::North], 3.2 + 0.01 * SpeedIndexes::North);
    ASSERT_FLOAT_EQ(tmp_cells[0][1][SpeedIndexes::NorthEast], 4.2 + 0.01 * SpeedIndexes::NorthEast);
    ASSERT_FLOAT_EQ(tmp_cells[0][1][SpeedIndexes::West], 2.3 + 0.01 * SpeedIndexes::West);
    ASSERT_FLOAT_EQ(tmp_cells[0][1][SpeedIndexes::Middle], 3.3 + 0.01 * SpeedIndexes::Middle);
    ASSERT_FLOAT_EQ(tmp_cells[0][1][SpeedIndexes::East], 4.3 + 0.01 * SpeedIndexes::East);
    ASSERT_FLOAT_EQ(tmp_cells[0][1][SpeedIndexes::SouthWest], 2.4 + 0.01 * SpeedIndexes::SouthWest);
    ASSERT_FLOAT_EQ(tmp_cells[0][1][SpeedIndexes::South], 3.4 + 0.01 * SpeedIndexes::South);
    ASSERT_FLOAT_EQ(tmp_cells[0][1][SpeedIndexes::SouthEast], 4.4 + 0.01 * SpeedIndexes::SouthEast);

    ASSERT_FLOAT_EQ(tmp_cells[0][2][SpeedIndexes::NorthWest], 3.2 + 0.01 * SpeedIndexes::NorthWest);
    ASSERT_FLOAT_EQ(tmp_cells[0][2][SpeedIndexes::North], 4.2 + 0.01 * SpeedIndexes::North);
    ASSERT_FLOAT_EQ(tmp_cells[0][2][SpeedIndexes::NorthEast], 5.2 + 0.01 * SpeedIndexes::NorthEast);
    ASSERT_FLOAT_EQ(tmp_cells[0][2][SpeedIndexes::West], 3.3 + 0.01 * SpeedIndexes::West);
    ASSERT_FLOAT_EQ(tmp_cells[0][2][SpeedIndexes::Middle], 4.3 + 0.01 * SpeedIndexes::Middle);
    ASSERT_FLOAT_EQ(tmp_cells[0][2][SpeedIndexes::East], 5.3 + 0.01 * SpeedIndexes::East);
    ASSERT_FLOAT_EQ(tmp_cells[0][2][SpeedIndexes::SouthWest], 3.4 + 0.01 * SpeedIndexes::SouthWest);
    ASSERT_FLOAT_EQ(tmp_cells[0][2][SpeedIndexes::South], 4.4 + 0.01 * SpeedIndexes::South);
    ASSERT_FLOAT_EQ(tmp_cells[0][2][SpeedIndexes::SouthEast], 5.4 + 0.01 * SpeedIndexes::SouthEast);

    ASSERT_FLOAT_EQ(tmp_cells[0][3][SpeedIndexes::NorthWest], 4.2 + 0.01 * SpeedIndexes::NorthWest);
    ASSERT_FLOAT_EQ(tmp_cells[0][3][SpeedIndexes::North], 5.2 + 0.01 * SpeedIndexes::North);
    ASSERT_FLOAT_EQ(tmp_cells[0][3][SpeedIndexes::NorthEast], 6.2 + 0.01 * SpeedIndexes::NorthEast);
    ASSERT_FLOAT_EQ(tmp_cells[0][3][SpeedIndexes::West], 4.3 + 0.01 * SpeedIndexes::West);
    ASSERT_FLOAT_EQ(tmp_cells[0][3][SpeedIndexes::Middle], 5.3 + 0.01 * SpeedIndexes::Middle);
    ASSERT_FLOAT_EQ(tmp_cells[0][3][SpeedIndexes::East], 6.3 + 0.01 * SpeedIndexes::East);
    ASSERT_FLOAT_EQ(tmp_cells[0][3][SpeedIndexes::SouthWest], 4.4 + 0.01 * SpeedIndexes::SouthWest);
    ASSERT_FLOAT_EQ(tmp_cells[0][3][SpeedIndexes::South], 5.4 + 0.01 * SpeedIndexes::South);
    ASSERT_FLOAT_EQ(tmp_cells[0][3][SpeedIndexes::SouthEast], 6.4 + 0.01 * SpeedIndexes::SouthEast);
}


TEST(propagate, testPropagateVertexTopRight) {
    const auto nx = 5u; // excluding halo
    const auto ny = 3u;
    auto tmp_cells = runPropagate();

    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][nx - 1][SpeedIndexes::NorthWest], 5.0 + 0.01 * SpeedIndexes::NorthWest);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][nx - 1][SpeedIndexes::North], 6.0 + 0.01 * SpeedIndexes::North);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][nx - 1][SpeedIndexes::NorthEast], 7.0 + 0.01 * SpeedIndexes::NorthEast);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][nx - 1][SpeedIndexes::West], 5.1 + 0.01 * SpeedIndexes::West);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][nx - 1][SpeedIndexes::Middle], 6.1 + 0.01 * SpeedIndexes::Middle);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][nx - 1][SpeedIndexes::East], 7.1 + 0.01 * SpeedIndexes::East);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][nx - 1][SpeedIndexes::SouthWest], 5.2 + 0.01 * SpeedIndexes::SouthWest);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][nx - 1][SpeedIndexes::South], 6.2 + 0.01 * SpeedIndexes::South);
    ASSERT_FLOAT_EQ(tmp_cells[ny - 1][nx - 1][SpeedIndexes::SouthEast], 7.2 + 0.01 * SpeedIndexes::SouthEast);


}


TEST(propagate, testPropagateVertexBottomLeft) {
    const auto nx = 5u; // excluding halo
    const auto ny = 3u;
    auto tmp_cells = runPropagate();

    ASSERT_FLOAT_EQ(tmp_cells[0][0][SpeedIndexes::NorthWest], 1.2 + 0.01 * SpeedIndexes::NorthWest);
    ASSERT_FLOAT_EQ(tmp_cells[0][0][SpeedIndexes::North], 2.2 + 0.01 * SpeedIndexes::North);
    ASSERT_FLOAT_EQ(tmp_cells[0][0][SpeedIndexes::NorthEast], 3.2 + 0.01 * SpeedIndexes::NorthEast);
    ASSERT_FLOAT_EQ(tmp_cells[0][0][SpeedIndexes::West], 1.3 + 0.01 * SpeedIndexes::West);
    ASSERT_FLOAT_EQ(tmp_cells[0][0][SpeedIndexes::Middle], 2.3 + 0.01 * SpeedIndexes::Middle);
    ASSERT_FLOAT_EQ(tmp_cells[0][0][SpeedIndexes::East], 3.3 + 0.01 * SpeedIndexes::East);
    ASSERT_FLOAT_EQ(tmp_cells[0][0][SpeedIndexes::SouthWest], 1.4 + 0.01 * SpeedIndexes::SouthWest);
    ASSERT_FLOAT_EQ(tmp_cells[0][0][SpeedIndexes::South], 2.4 + 0.01 * SpeedIndexes::South);
    ASSERT_FLOAT_EQ(tmp_cells[0][0][SpeedIndexes::SouthEast], 3.4 + 0.01 * SpeedIndexes::SouthEast);
}


TEST(propagate, testPropagateVertexBottomRight) {
    const auto nx = 5u; // excluding halo
    const auto ny = 3u;
    auto tmp_cells = runPropagate();

    ASSERT_FLOAT_EQ(tmp_cells[0][nx - 1][SpeedIndexes::NorthWest], 5.2 + 0.01 * SpeedIndexes::NorthWest);
    ASSERT_FLOAT_EQ(tmp_cells[0][nx - 1][SpeedIndexes::North], 6.2 + 0.01 * SpeedIndexes::North);
    ASSERT_FLOAT_EQ(tmp_cells[0][nx - 1][SpeedIndexes::NorthEast], 7.2 + 0.01 * SpeedIndexes::NorthEast);
    ASSERT_FLOAT_EQ(tmp_cells[0][nx - 1][SpeedIndexes::West], 5.3 + 0.01 * SpeedIndexes::West);
    ASSERT_FLOAT_EQ(tmp_cells[0][nx - 1][SpeedIndexes::Middle], 6.3 + 0.01 * SpeedIndexes::Middle);
    ASSERT_FLOAT_EQ(tmp_cells[0][nx - 1][SpeedIndexes::East], 7.3 + 0.01 * SpeedIndexes::East);
    ASSERT_FLOAT_EQ(tmp_cells[0][nx - 1][SpeedIndexes::SouthWest], 5.4 + 0.01 * SpeedIndexes::SouthWest);
    ASSERT_FLOAT_EQ(tmp_cells[0][nx - 1][SpeedIndexes::South], 6.4 + 0.01 * SpeedIndexes::South);
    ASSERT_FLOAT_EQ(tmp_cells[0][nx - 1][SpeedIndexes::SouthEast], 7.4 + 0.01 * SpeedIndexes::SouthEast);
}

TEST(propagate, testPropagateVertexLeft) {
    auto tmp_cells = runPropagate();

    ASSERT_FLOAT_EQ(tmp_cells[1][0][SpeedIndexes::NorthWest], 1.1 + 0.01 * SpeedIndexes::NorthWest);
    ASSERT_FLOAT_EQ(tmp_cells[1][0][SpeedIndexes::North], 2.1 + 0.01 * SpeedIndexes::North);
    ASSERT_FLOAT_EQ(tmp_cells[1][0][SpeedIndexes::NorthEast], 3.1 + 0.01 * SpeedIndexes::NorthEast);
    ASSERT_FLOAT_EQ(tmp_cells[1][0][SpeedIndexes::West], 1.2 + 0.01 * SpeedIndexes::West);
    ASSERT_FLOAT_EQ(tmp_cells[1][0][SpeedIndexes::Middle], 2.2 + 0.01 * SpeedIndexes::Middle);
    ASSERT_FLOAT_EQ(tmp_cells[1][0][SpeedIndexes::East], 3.2 + 0.01 * SpeedIndexes::East);
    ASSERT_FLOAT_EQ(tmp_cells[1][0][SpeedIndexes::SouthWest], 1.3 + 0.01 * SpeedIndexes::SouthWest);
    ASSERT_FLOAT_EQ(tmp_cells[1][0][SpeedIndexes::South], 2.3 + 0.01 * SpeedIndexes::South);
    ASSERT_FLOAT_EQ(tmp_cells[1][0][SpeedIndexes::SouthEast], 3.3 + 0.01 * SpeedIndexes::SouthEast);
}


TEST(propagate, testPropagateVertexRight) {
    auto tmp_cells = runPropagate();
    auto nx = 5;

    ASSERT_FLOAT_EQ(tmp_cells[1][nx - 1][SpeedIndexes::NorthWest], 5.1 + 0.01 * SpeedIndexes::NorthWest);
    ASSERT_FLOAT_EQ(tmp_cells[1][nx - 1][SpeedIndexes::North], 6.1 + 0.01 * SpeedIndexes::North);
    ASSERT_FLOAT_EQ(tmp_cells[1][nx - 1][SpeedIndexes::NorthEast], 7.1 + 0.01 * SpeedIndexes::NorthEast);
    ASSERT_FLOAT_EQ(tmp_cells[1][nx - 1][SpeedIndexes::West], 5.2 + 0.01 * SpeedIndexes::West);
    ASSERT_FLOAT_EQ(tmp_cells[1][nx - 1][SpeedIndexes::Middle], 6.2 + 0.01 * SpeedIndexes::Middle);
    ASSERT_FLOAT_EQ(tmp_cells[1][nx - 1][SpeedIndexes::East], 7.2 + 0.01 * SpeedIndexes::East);
    ASSERT_FLOAT_EQ(tmp_cells[1][nx - 1][SpeedIndexes::SouthWest], 5.3 + 0.01 * SpeedIndexes::SouthWest);
    ASSERT_FLOAT_EQ(tmp_cells[1][nx - 1][SpeedIndexes::South], 6.3 + 0.01 * SpeedIndexes::South);
    ASSERT_FLOAT_EQ(tmp_cells[1][nx - 1][SpeedIndexes::SouthEast], 7.3 + 0.01 * SpeedIndexes::SouthEast);
}
