// import numpy as np
// import matplotlib.pyplot as plt
// from matplotlib import cm
// from tqdm import tqdm

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

using namespace poplar;
using namespace poplar::program;
using namespace popops;


// %matplotlib inline
// %config InlineBackend.figure_format='retina'

// # Flow definition
// maxIter = 200 * 1000
// Re = 220.0 # Reynolds number
// nx, ny = 420, 180 # Number of lattice nodes
// ly = ny - 1 # Height of the domain of the lattice units
// cx, cy, r = nx //4, ny //2, ny//9 # Coordinates of the cylinder
// sx, sy, sw = 2 * nx //3, ny // 3, ny // 7 # Coordinates of square obstacle
// uLB = 0.04 # Velocity of the lattice units
// nulb = uLB * r/ Re # Viscosity in lattice units
// omega = 1 / (3 * nulb + 0.5) # Relaxation Parameter

struct {
    // const unsigned maxIter = 20 * 1000;
    const unsigned maxIter = 2;
    const float Re = 220.0;     // Reynolds number
    const unsigned nx = 128;    // Number of lattice nodes
    const unsigned ny = 127;    // Number of lattice nodes
    const unsigned ly = ny - 1; //  Height of the domain of the lattice units
    const unsigned cx = nx / 4;
    const unsigned cy = ny / 2;
    const unsigned r = ny / 9; //  Coordinates of the cylinder obstacle
    const unsigned sx = 2 * nx / 3;
    const unsigned sy = ny / 3;
    const unsigned sw = ny / 7;               // Coordinates of square obstacle
    const float uLB = 0.04;                   // Velocity of the lattice units
    const float nulb = uLB * (float) r / Re;          // Viscosity in lattice units
    const float omega = 1.0f / (3.0f * nulb + 0.5f); //  Relaxation Parameter
    const float accel = 1;                    // Density redistribution
    const float density = 1;                  // Density per link
} Constants;

// def macroscopic(fin):
//     rho = np.sum(fin, axis=0)
//     u = np.zeros((2, nx, ny))
//     for i in range(9):
//         u[0,:,:] += v[i,0] * fin[i,:,:]
//         u[1,:,:] += v[i,1] * fin[i,:,:]
//     u /= rho
//     return rho, u
auto macroscopic(Graph &graph, std::map<std::string, Tensor> &tensors,
                 const std::vector<std::tuple<int, int>> &v) -> Program {
    std::cerr << __FUNCTION__ << std::endl;

    Sequence prog;

    reduceWithOutput(graph, tensors["fin"], tensors["rho"], {0},
                     {Operation::ADD}, prog, "rho=sum(fin, axis=0)");
    mulInPlace(graph, tensors["u"], tensors["0"], prog);
    for (auto i = 0u; i < 9; i++) {
        auto vi0 = tensors[std::to_string(std::get<0>(v[i]))];
        auto vi1 = tensors[std::to_string(std::get<1>(v[i]))];
        // TODO could parallalise all 9 thingies by unrolling, storing in partial and reducing
        const auto a = mul(graph, vi0, tensors["fin"][i], prog, "v[i,0] * fin[i,:,:]");
        const auto b = mul(graph, vi1, tensors["fin"][i], prog, "v[i,1] * fin[i,:,:]");
        addInPlace(graph, tensors["u"][0], a, prog, "u[0,:,:] += v[i,0] * fin[i,:,:]");
        addInPlace(graph, tensors["u"][1], b, prog, "u[1,:,:] += v[i,1] * fin[i,:,:]");
    }
    divInPlace(graph, tensors["u"], tensors["rho"], prog, "u /= rho");
    return std::move(prog);
}

// def equilibrium(rho, u):
//     usqr= 3/2 * (u[0]**2 + u[1]**2)
//     feq = np.zeros((9,nx,ny))
//     for i in range(9):
//         cu = 3 * (v[i,0]*u[0,:,:] + v[i,1]*u[1,:,:])
//         feq[i,:,:] = rho * t[i] * (1+ cu + 0.5 * cu ** 2 - usqr)
//     return feq
auto equilibrium(Graph &graph, std::map<std::string, Tensor> &tensors,
                 const std::vector<std::tuple<int, int>> &v) -> Program {
    std::cerr << __FUNCTION__ << std::endl;

    Sequence prog;

    auto usqr = mul(graph,
                    tensors["3/2"],
                    add(
                            graph,
                            square(graph, tensors["u"][0], prog, "u[0]**2"),
                            square(graph, tensors["u"][1], prog, "u[1]**2"),
                            prog),
                    prog,
                    "usqr=3/2 * (u[0]**2 + u[1]**2)");

    mulInPlace(graph, tensors["feq"], tensors["0"], prog);

    for (auto i = 0u; i < 9; i++) // Can probably paralallise & reduce this
    {
        auto cu = mul(
                graph,
                tensors["3"],
                add(
                        graph,
                        mul(
                                graph,
                                tensors[std::to_string(std::get<0>(v[i]))],
                                tensors["u"][0],
                                prog,
                                "v[i,0]*u[0,:,:]"),
                        mul(
                                graph,
                                tensors[std::to_string(std::get<1>(v[i]))],
                                tensors["u"][1],
                                prog,
                                "v[i,1]*u[1,:,:]"),
                        prog,
                        "(v[i,0]*u[0,:,:] + v[i,1]*u[1,:,:])"),
                prog,
                "cu = 3 * (v[i,0]*u[0,:,:] + v[i,1]*u[1,:,:])");

        auto cusqr = square(graph, cu, prog, "cu**2");
        auto half_cusqr = mul(graph, cusqr, tensors["0.5"], prog, "0.5 * cu**2");
        auto half_cusqr_minus_usqr = sub(graph, half_cusqr, usqr, prog, "0.5 * cu ** 2 - usqr");
        auto one_plus_cu = add(graph, tensors["1"], cu, prog, "1+cu");
        auto all_in_par = add(graph, one_plus_cu, half_cusqr_minus_usqr, prog, "(1+ cu + 0.5 * cu ** 2 - usqr)");
        auto rho_ti = mul(graph, tensors["rho"], tensors["t"][i], prog, "rho * t[i]");
        auto feqi = mul(graph, rho_ti, all_in_par, prog, "feq[i,:,:] = rho * t[i] * (1+ cu + 0.5 * cu ** 2 - usqr)");
        prog.add(Copy(feqi, tensors["feq"][i], "feq[i,:,:] = rho * t[i] * (1+ cu + 0.5 * cu ** 2 - usqr)"));
    }
    return std::move(prog);
}

// def initial_velocity_fun(d, x, y):
//     return (1 - d) * uLB * (1 + 1e-4 * np.sin(y/ly * 2 * np.pi))
//
// vel = np.fromfunction(initial_velocity_fun, (2, nx, ny))
auto initialiseVelocityTensor(Graph &graph, std::map<std::string, Tensor> &tensors) {
    auto vel = std::vector<float>(2 * Constants.nx * Constants.ny, {});

    for (auto d = 0u; d < 2u; d++)
        for (auto x = 0U; x < Constants.nx; x++)
            for (auto y = 0u; y < Constants.ny; y++)
                vel[d * (x * y) + x * y + y] =
                        (1.0f - d) * Constants.uLB * (1 + 1e-4 * sin(y / Constants.ly * 2 * M_PI));

    tensors["vel"] = graph.addConstant(FLOAT, {2, Constants.nx, Constants.ny}, vel.data(), "vel");
    poputil::mapTensorLinearly(graph, tensors["vel"]);
}

auto randomTile(const unsigned numTiles) -> unsigned int {
    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned> distribution(0, 1216);
    return distribution(generator);
}

auto addScalarContants(Graph &graph, std::map<std::string, Tensor> &tensors,
                       const std::map<std::string, float> &thingsToAdd, const unsigned numTiles) {
    for (auto const &[name, value] : thingsToAdd) {
        tensors[name] = graph.addConstant(FLOAT, {}, value, name);
        // We map to random vertex because they'll be accessed from everywhere
        graph.setTileMapping(tensors[name], randomTile(numTiles));
    }
}

auto incrementTimestep(Graph &graph, std::map<std::string, Tensor> &tensors) -> Program {
    std::cerr << __FUNCTION__ << std::endl;

    Sequence s;
    popops::addInPlace(graph, tensors["timestep"], tensors["1u"], s, "timestep++");
    return std::move(s);
}


auto inflowCondition(Graph &graph, std::map<std::string, Tensor> &tensors) -> Program {
    std::cerr << __FUNCTION__ << std::endl;

    //     # Left wall: inflow condition
    //     u[:,0,:] = vel[:,0,:]
    //     rho[0,:] = 1/(1-u[0,0,:]) * (np.sum(fin[col2, 0, :], axis=0) + 2 * np.sum(fin[col3,0,:], axis=0))
    Sequence s;

    s.add(Copy(tensors["vel"].slice(0, 1, 1), tensors["u"].slice(0, 1, 1), "u[:,0,:] = vel[:,0,:]"));

    // col2 is 3,4,5, col3 is 6,7,8
    auto sum_fin_345 = reduce(graph, tensors["fin"].slice(3, 6, 0)[0], FLOAT, {0},
                              popops::ReduceParams(popops::Operation::ADD), s, "sum(fin[col2, 0, :], axis=0)");
    auto sum_fin_678 = popops::reduce(graph, tensors["fin"].slice(7, 9, 0)[0], FLOAT, {0},
                                      popops::ReduceParams(popops::Operation::ADD), s, "sum(fin[col3, 0, :], axis=0)");
    auto one_min_u00 = popops::sub(graph, tensors["1"], tensors["u"][0][0], s, "1-u[0,0,:]");
    popops::invInPlace(graph, one_min_u00, s, "1/(1-u[0,0,:])");
    popops::scaledAddTo(graph, sum_fin_345, sum_fin_678, tensors["2"], s,
                        "sum(fin[col2, 0, :], axis=0) + 2*sum(fin[col3,0,:], axis=0)");
    popops::mulInPlace(graph, sum_fin_345, one_min_u00, s,
                       "1/(1-u[0,0,:]) * (np.sum(fin[col2, 0, :], axis=0) + 2 * np.sum(fin[col3,0,:], axis=0))");
    s.add(Copy(sum_fin_345, tensors["rho"][0]));

    return std::move(s);
}

auto outflowCondition(Graph &graph, std::map<std::string, Tensor> &tensors,
                      const std::vector<std::tuple<int, int>> &v) -> Program {
    std::cerr << __FUNCTION__ << std::endl;

    //     # Right wall: outflow condition
    //     fin[col3, -1, :] = fin[col3, -2, :]
    //     rho, u = macroscopic(fin)
    Sequence s;
    //col3 is 6,7,8

    const auto last_idx = tensors["fin"].dim(1) - 1;
    const auto penultimate_idx = last_idx - 1;
    assert(penultimate_idx > 0);
    auto tmp1 = tensors["fin"].slice(6, 9, 0).slice(last_idx, last_idx + 1, 1);
    auto tmp2 = tensors["fin"].slice(6, 9, 0).slice(penultimate_idx, last_idx, 1);
    // std::cout << tmp1.dim(0) << " " << tmp1.dim(1) << " " << tmp1.dim(2) << std::endl;
    // std::cout << tmp2.dim(0) << " " << tmp2.dim(1) << " " << tmp1.dim(2) << std::endl;
    add(graph, tmp1, tmp2, s, "tmp1+tmp2");
    auto x = graph.addVariable(FLOAT, {3, 1, Constants.ny}, poplar::VariableMappingMethod::LINEAR, "x");

    s.add(Copy(tmp2, x, "fin[[6,7,8]], -1, :] = fin[[6,7,8], -2, :]"));
    s.add(Copy(x, tmp1, "fin[[6,7,8]], -1, :] = fin[[6,7,8], -2, :]"));
    //s.add(Copy(tmp2, tmp1, "fin[[6,7,8]], -1, :] = fin[[6,7,8], -2, :]")); // And yet this line has an exception, don't know why
    // TODO Raise a bug. This works on a smaller tensor?!

    s.add(macroscopic(graph, tensors, v));

    return std::move(s);
}


auto
streaming(Graph &graph, std::map<std::string, Tensor> &tensors, const std::vector<std::tuple<int, int>> &v) -> Program {
    // v = np.array([ [1, 1], [1, 0],  [1,-1], [0,1], [0,0],
    //             [0,-1], [-1, 1], [-1, 0], [-1,-1]])
    //     for i in range(9):
    //         fin[i,:,:] = np.roll(np.roll(fout[i,:,:], v[i,0], axis=0), v[i,1], axis=1)

    // This is the one with 9 elems. We rolling all the elements around
    //TODO feature request for roll in Poplar
    Sequence s;
    size_t i = 0;
    for (auto &[x, y] : v) {
        auto src = tensors["fout"][i];
        auto dst = tensors["fin"][i];
        s.add(doubleRolledCopy(graph, src, dst, x, y));
        i++;
    }
    return std::move(s);
}

auto equilibriumStep(Graph &graph, std::map<std::string, Tensor> &tensors,
                     const std::vector<std::tuple<int, int>> &v) -> Program {
    //     feq = equilibrium(rho, u)
    //     fin[[0,1,2],0,:] = feq[[0,1,2],0,:] + fin[[8,7,6], 0, :] - feq[[8,7,6],0,:]
    Sequence s;
    s.add(equilibrium(graph, tensors, v));
    auto tmp = sub(graph, tensors["fin"].slice(6, 9, 0)[0], tensors["feq"].slice(6, 9, 0)[0], s,
                   "fin[[6,7,8], 0, :] - feq[[6,7,8],0,:]");
    auto reversed = tmp.reverse(0);
    s.add(Copy(tensors["feq"].slice(0, 3, 0)[0], tensors["fin"].slice(0, 3, 0)[0]));
    addInPlace(graph, tensors["fin"].slice(0, 3, 0)[0], reversed, s,
               "fin[[0,1,2],0,:] = feq[[0,1,2],0,:] + fin[[8,7,6], 0, :] - feq[[8,7,6],0,:]");
    return std::move(s);
}

auto collision(Graph &graph, std::map<std::string, Tensor> &tensors) -> Program {
    //     fout = fin - omega * (fin - feq)
    Sequence s;
    auto fin_min_feq = sub(graph, tensors["fin"], tensors["feq"], s, "(fin - feq)");
    s.add(Copy(tensors["fin"], tensors["fout"]));
    scaledAddTo(graph, tensors["fout"], fin_min_feq, tensors["-omega"], s, "fout=fin - omega * (fin - feq)");
    return std::move(s);
}

Program bounceBack(Graph &graph, std::map<std::string, Tensor> &tensors) {
    //     for i in range(9):
    //         fout[i, obstacle] = fin[8-i, obstacle]
    Sequence s;
    // for (auto i = 0u; i < 9; i++)
    // {
    //     auto tmp1 = tensors["fout"][i];
    //     auto tmp2 = tensors["fin"][8 - i];
    //     popops::select(graph, tmp1, tmp2, tensors["obstacle"], s, "fout[i, obstacle] = fin[8-i, obstacle]");
    // }

    // actually we can do this in one 3x3 tensor op!
    poputil::broadcastToMatch(tensors["obstacle"], {9, Constants.nx, Constants.ny}); //TODO: just do this on creation
    popops::select(graph, tensors["fout"], tensors["fin"].reverse(0), tensors["obstacle"], s,
                   "fout[i, obstacle] = fin[8-i, obstacle]");
    return std::move(s);
}


auto initialise(Graph &graph, std::map<std::string, Tensor> &tensors,
                const std::vector<std::tuple<int, int>> &v) -> Program {
    Sequence s;
    // fin = equilibrium(1, vel)
    s.add(Copy(tensors["vel"], tensors["u"]));
    // Set rho to 1 for first call
    mul(graph, tensors["rho"], tensors["0"], s, "rho=0");
    addInPlace(graph, tensors["rho"], tensors["1"], s, "rho+=1");
    s.add(equilibrium(graph, tensors, v));

    return std::move(s);
}

bool isObstacle(unsigned x, unsigned y) {

    auto isCircle = (x - Constants.cx) * (x - Constants.cx) + (y - Constants.cy) * (y - Constants.cy) <
                    Constants.r * Constants.r;
    auto isSquare = (x > (Constants.sx - Constants.sw) / 2) && (x < (Constants.sx + Constants.sw) / 2) &&
                    (y > (Constants.sy - Constants.sw) / 2) && (y < (Constants.sy + Constants.sw) / 2);
    return isSquare | isCircle;
}

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

    double total_compute_time = 0.0;
    std::chrono::high_resolution_clock::time_point tic, toc;
    auto device = lbm::getIpuModel();
    //auto device = getIpuDevice();
    if (!device.has_value()) {
        return EXIT_FAILURE;
    }

    auto tensors = std::map<std::string, Tensor>{};

    //------
    std::cerr << "Building computational graph" << std::endl;
    tic = std::chrono::high_resolution_clock::now();

    Graph graph(device.value().getTarget());
    popops::addCodelets(graph);

    // # Lattice constants
    // v = np.array([ [1, 1], [1, 0],  [1,-1], [0,1], [0,0],
    //             [0,-1], [-1, 1], [-1, 0], [-1,-1]])
    // t = np.array([1/36, 1/9, 1/36, 1/9, 4/9, 1/9, 1/36, 1/9, 1/36])
    auto v = std::vector<std::tuple<int, int>>{
            {{1, 1}, {1, 0}, {1, -1}, {0, 1}, {0, 0}, {0, -1}, {-1, 1}, {-1, 0}, {-1, -1}}}; // the directions the dimensions mean

    const std::vector<float> tInit = {
            Constants.density * Constants.accel / 36,
            Constants.density * Constants.accel / 9,
            Constants.density * Constants.accel / 36,
            Constants.density * Constants.accel / 9,
            4.0f * Constants.density * Constants.accel / 9,
            Constants.density * Constants.accel / 9,
            Constants.density * Constants.accel / 36,
            Constants.density * Constants.accel / 9,
            Constants.density * Constants.accel / 36};
    tensors["t"] = graph.addConstant(FLOAT, {9}, tInit.data(), "t");
    poputil::mapTensorLinearly(graph, tensors["t"]);

    // u = np.zeros((2, nx, ny))
    tensors["u"] = graph.addVariable(FLOAT, {2, Constants.nx, Constants.ny}, "u");
    poputil::mapTensorLinearly(graph, tensors["u"]);

    // feq = np.zeros((9,nx,ny))
    for (auto name : {"fin", "feq", "fout"}) // TODO consider using popops::createSliceableTensor
    {
        tensors[name] = graph.addVariable(FLOAT, {9, Constants.nx, Constants.ny}, name);
        poputil::mapTensorLinearly(graph, tensors[name]);
    }

    auto obstacle_array = std::unique_ptr<bool[]>(new bool[Constants.nx * Constants.ny]());
    for (auto i = 0u; i < Constants.nx; i++) {
        for (auto j = 0u; j < Constants.ny; j++) {
            obstacle_array[i * Constants.ny + j] = isObstacle(i, j);
        }
    }

    auto output_array = std::unique_ptr<float[]>(new float[2 * Constants.nx * Constants.ny]());

    tensors["obstacle"] = graph.addConstant(BOOL, {Constants.nx, Constants.ny}, obstacle_array.get(), "obstacle");
    poputil::mapTensorLinearly(graph, tensors["obstacle"]);

    tensors["rho"] = graph.addVariable(FLOAT, {Constants.nx, Constants.ny}, "rho");
    poputil::mapTensorLinearly(graph, tensors["rho"]);

    tensors["1u"] = graph.addConstant(UNSIGNED_INT, {}, 1u, "1u");
    graph.setTileMapping(tensors["1u"], rand() % graph.getTarget().getNumTiles());

    tensors["timestep"] = graph.addVariable(UNSIGNED_INT, {}, "timestep");
    graph.setTileMapping(tensors["timestep"], 5);
    graph.setInitialValue(tensors["timestep"], 0u);

    addScalarContants(
            graph, tensors,
            {{"-1",     -1.0f},
             {"0",      0.0f},
             {"0.5",    0.5f},
             {"1",      1.0f},
             {"2",      2.0f},
             {"3",      3.0f},
             {"3/2",    3.0 / 2.0f},
             {"-omega", -Constants.omega}},
            graph.getTarget().getNumTiles());

    Sequence prog;
    constexpr auto LOOPS_BEFORE_READ = 2;

    // auto equilibriumFn = graph.addFunction(equilibrium(graph, tensors));
    // auto macroscopicFn = graph.addFunction(macroscopic(graph, tensors));
    // auto incrementTimestepFn = graph.addFunction(incrementTimestep(graph, tensors));

    initialiseVelocityTensor(graph, tensors);

    prog.add(

            Repeat(LOOPS_BEFORE_READ,
                   Sequence(
                           outflowCondition(graph, tensors, v),
                           inflowCondition(graph, tensors),
                           equilibriumStep(graph, tensors, v),
                           collision(graph, tensors),
                           bounceBack(graph, tensors),
                           streaming(graph, tensors, v),
                           incrementTimestep(graph, tensors))));

    toc = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic).count() * 1000;
    std::cerr << "Took " << std::right << std::setw(12) << std::setprecision(5) << diff << "ms" << std::endl;

    //-----

    std::cerr << "Creating engine and loading computational graph" << std::endl;
    tic = std::chrono::high_resolution_clock::now();

    auto outStream = graph.addDeviceToHostFIFO("u", FLOAT, 2 * Constants.nx * Constants.ny);

    auto engine = lbm::createDebugEngine(graph, {initialise(graph, tensors, v), prog, Copy(tensors["u"], outStream)});
    engine.connectStream(outStream, output_array.get());
    std::cerr << "Loading..." << std::endl;

    engine.load(device.value());

    toc = std::chrono::high_resolution_clock::now();

    diff = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic).count();
    std::cerr << "Took " << std::right << std::setw(12) << std::setprecision(5) << diff << "s" << std::endl;

    std::cerr << "Running initiatilisation step ";
    tic = std::chrono::high_resolution_clock::now();

    engine.run(0);
    toc = std::chrono::high_resolution_clock::now();
    diff = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic).count();
    std::cerr << "took " << std::right << std::setw(12) << std::setprecision(5) << diff << "s" << std::endl;
    total_compute_time += diff;

    // ---
    for (unsigned l = 0; l < Constants.maxIter / LOOPS_BEFORE_READ; l++) {
        std::cerr << "Running " << LOOPS_BEFORE_READ << " iters ";
        tic = std::chrono::high_resolution_clock::now();
        engine.run(1);
        toc = std::chrono::high_resolution_clock::now();
        diff = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic).count();
        std::cerr << "took " << std::right << std::setw(12) << std::setprecision(5) << diff << "s" << std::endl;
        total_compute_time += diff;

        std::cerr << "Copy to host ";
        tic = std::chrono::high_resolution_clock::now();
        engine.run(2);
        toc = std::chrono::high_resolution_clock::now();
        diff = std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic).count();
        std::cerr << "took " << std::right << std::setw(12) << std::setprecision(5) << diff << "s" << std::endl;
    }

    lbm::captureProfileInfo(engine);

    engine.printProfileSummary(std::cout,
                               OptionFlags{{"showExecutionSteps", "false"}});

    std::cerr << "Total compute time was  " << std::right << std::setw(12) << std::setprecision(5) << total_compute_time
              << "s" << std::endl;

    return EXIT_SUCCESS;
}

// # Lattice constants
// col1 = np.array([0,1,2])
// col2 = np.array([3,4,5])
// col3 = np.array([6,7,8])

// def equilibrium(rho, u):
//     usqr= 3/2 * (u[0]**2 + u[1]**2)
//     feq = np.zeros((9,nx,ny))
//     for i in range(9):
//         cu = 3 * (v[i,0]*u[0,:,:] + v[i,1]*u[1,:,:])
//         feq[i,:,:] = rho * t[i] * (1+ cu + 0.5 * cu ** 2 - usqr)
//     return feq

// def obstacle_fun(x, y):
//     circle =  (x - cx) ** 2 + (y - cy) ** 2 < r ** 2
//     square = (x > sx - sw/2) & (x  <  sx + sw / 2) & (y  >  sy - sw / 2) & (y  <  sy + sw / 2)
//     return square | circle

// obstacle = np.fromfunction(obstacle_fun, (nx, ny))

// def initial_velocity_fun(d, x, y):
//     return (1 - d) * uLB * (1 + 1e-4 * np.sin(y/ly * 2 * np.pi))

// vel = np.fromfunction(initial_velocity_fun, (2, nx, ny))

// fin = equilibrium(1, vel)

// for time in tqdm(range(maxIter)):
//     # Right wall: outflow condition
//     fin[col3, -1, :] = fin[col3, -2, :]
//     rho, u = macroscopic(fin)

//     # Left wall: inflow condition
//     u[:,0,:] = vel[:,0,:]
//     rho[0,:] = 1/(1-u[0,0,:]) * (np.sum(fin[col2, 0, :], axis=0) + 2 * np.sum(fin[col3,0,:], axis=0))

//     # Equilibrium
//     feq = equilibrium(rho, u)
//     fin[[0,1,2],0,:] = feq[[0,1,2],0,:] + fin[[8,7,6], 0, :] - feq[[8,7,6],0,:]

//     # Collision
//     fout = fin - omega * (fin - feq)

//     # Bounce-back for obstacle
//     for i in range(9):
//         fout[i, obstacle] = fin[8-i, obstacle]

//     # Streaming step
//     for i in range(9):
//         fin[i,:,:] = np.roll(np.roll(fout[i,:,:], v[i,0], axis=0), v[i,1], axis=1)

//     if (time%100 == 0):
//         plt.clf()
//         plt.imshow(np.sqrt(u[0] ** 2 + u[1] ** 2).transpose(), cmap=cm.Spectral)
//         plt.axis('off')
//         plt.savefig("vel.{0:04d}.png".format(time // 100))
