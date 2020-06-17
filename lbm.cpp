// import numpy as np
// import matplotlib.pyplot as plt
// from matplotlib import cm
// from tqdm import tqdm

#include <cstdlib>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ScaledAdd.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poplar/DeviceManager.hpp>
#include <iomanip>
#include <iostream>
#include <poplar/Program.hpp>
#include <popops/Reduce.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>

using namespace poplar;
using namespace poplar::program;


const OptionFlags POPLAR_ENGINE_OPTIONS{
        {"target.saveArchive",                "archive.a"},
        {"debug.instrument",                  "true"},
        {"debug.instrumentCompute",           "true"},
        {"debug.loweredVarDumpFile",          "vars.capnp"},
        {"debug.instrumentControlFlow",       "true"},
        {"debug.computeInstrumentationLevel", "tile"}
};


// %matplotlib inline
// %config InlineBackend.figure_format='retina'

poplar::Device getIpuModel()
{
    poplar::IPUModel ipuModel;
    ipuModel.numIPUs = 1;
    ipuModel.tilesPerIPU = 1216;
    return ipuModel.createDevice();
}

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

struct
{
    const unsigned maxIter = 200 * 1000;
    const float Re = 220.0;     // Reynolds number
    const unsigned nx = 420;    // Number of lattice nodes
    const unsigned ny = 180;    // Number of lattice nodes
    const unsigned ly = ny - 1; //  Height of the domain of the lattice units
    const unsigned cx = nx / 4;
    const unsigned cy = ny / 2;
    const unsigned r = ny / 9; //  Coordinates of the cylinder obstacle
    const unsigned sx = 2 * nx / 3;
    const unsigned sy = ny / 3;
    const unsigned sw = ny / 7;               // Coordinates of square obstacle
    const float uLB = 0.04;                   // Velocity of the lattice units
    const float nulb = uLB * r / Re;          // Viscosity in lattice units
    const float omega = 1 / (3 * nulb + 0.5); //  Relaxation Parameter
} Constants;

// def macroscopic(fin):
//     rho = np.sum(fin, axis=0)
//     u = np.zeros((2, nx, ny))
//     for i in range(9):
//         u[0,:,:] += v[i,0] * fin[i,:,:]
//         u[1,:,:] += v[i,1] * fin[i,:,:]
//     u /= rho
//     return rho, u
Program macroscopic(Graph &graph, std::map<std::string, Tensor> &tensors)
{
    program::Sequence prog;

    popops::reduceWithOutput(graph, tensors["fin"], tensors["rho"], {0},
                             popops::ReduceParams(popops::Operation::ADD), prog, "rho=sum(fin, axis=0)");
    popops::mulInPlace(graph, tensors["u"], tensors["0"], prog);
    for (auto i = 0u; i < 9; i++)
    {
        // TODO could parallalise all 9 thingies by unrolling, storing in partial and reducing
        popops::addInPlace(graph, tensors["u"][0], popops::mul(graph, tensors["v"][i][0], tensors["fin"][i], prog), prog);
        popops::addInPlace(graph, tensors["u"][1], popops::mul(graph, tensors["v"][i][1], tensors["fin"][i], prog), prog);
    }
    popops::divInPlace(graph, tensors["u"], tensors["rho"], prog);
    return prog;
 }

// def equilibrium(rho, u):
//     usqr= 3/2 * (u[0]**2 + u[1]**2)
//     feq = np.zeros((9,nx,ny))
//     for i in range(9):
//         cu = 3 * (v[i,0]*u[0,:,:] + v[i,1]*u[1,:,:])
//         feq[i,:,:] = rho * t[i] * (1+ cu + 0.5 * cu ** 2 - usqr)
//     return feq
Program equilibrium(Graph &graph, std::map<std::string, Tensor> &tensors)
{
    program::Sequence prog;

    auto usqr = popops::mul(graph,
                            tensors["3/2"],
                            popops::add(
                                graph,
                                popops::square(graph, tensors["u"][0], prog, "u[0]**2"),
                                popops::square(graph, tensors["u"][1], prog, "u[1]**2"),
                                prog),
                            prog,
                            "usqr=3/2 * (u[0]**2 + u[1]**2)");

    popops::mulInPlace(graph, tensors["feq"], tensors["0"], prog);

    for (auto i = 0u; i < 9; i++) // Can probably paralallise & reduce this
    {
        auto cu = popops::mul(
            graph,
            tensors["3"],
            popops::add(
                graph,
                popops::mul(
                    graph,
                    tensors["v"][i][0],
                    tensors["u"][0],
                    prog,
                    "v[i,0]*u[0,:,:]"),
                popops::mul(
                    graph,
                    tensors["v"][i][1],
                    tensors["u"][1],
                    prog,
                    "v[i,1]*u[1,:,:]"),
                prog,
                "(v[i,0]*u[0,:,:] + v[i,1]*u[1,:,:])"),
            prog,
            "cu = 3 * (v[i,0]*u[0,:,:] + v[i,1]*u[1,:,:])");

        auto cusqr = popops::square(graph, cu, prog, "cu**2");
        auto half_cusqr = popops::mul(graph, cusqr, tensors["0.5"], prog, "0.5 * cu**2");
        auto half_cusqr_minus_usqr = popops::sub(graph, half_cusqr, usqr, prog, "0.5 * cu ** 2 - usqr");
        auto one_plus_cu = popops::add(graph, tensors["1"], cu, prog, "1+cu");
        auto all_in_par = popops::add(graph, one_plus_cu, half_cusqr_minus_usqr, prog, "(1+ cu + 0.5 * cu ** 2 - usqr)");
        auto rho_ti = popops::add(graph, tensors["rho"], tensors["t"][i], prog, "rho * t[i]");
        auto feqi = popops::mul(graph, rho_ti, all_in_par, prog, "feq[i,:,:] = rho * t[i] * (1+ cu + 0.5 * cu ** 2 - usqr)");
        Copy(feqi, tensors["feq"][i], "feq[i,:,:] = rho * t[i] * (1+ cu + 0.5 * cu ** 2 - usqr)");
    }
    return prog;
}

// def initial_velocity_fun(d, x, y):
//     return (1 - d) * uLB * (1 + 1e-4 * np.sin(y/ly * 2 * np.pi))
//
// vel = np.fromfunction(initial_velocity_fun, (2, nx, ny))
void initialVelocity(Graph &graph, std::map<std::string, Tensor> &tensors)
{
    auto vel = std::vector<float>(2 * Constants.nx * Constants.ny, {});
    
    for (auto d = 0u; d < 2u; d++)
        for (auto x = 0U; x < Constants.nx; x++)
            for (auto y = 0u; y < Constants.ny; y++)
                vel[d * (x * y) + x * y + y] = (1 - d) * Constants.uLB * (1 + 1e-4 * sin(y / Constants.ly * 2 * M_PI));

    tensors["vel"] = graph.addConstant(FLOAT, {2, Constants.nx, Constants.ny}, vel.data(), "vel");
    poputil::mapTensorLinearly(graph, tensors["vel"]);

}

void addScalarContants(Graph &graph, std::map<std::string, Tensor> &tensors, std::map<std::string, float> thingsToAdd)
{
    for (auto const &[name, value] : thingsToAdd)
    {
        tensors[name] = graph.addConstant(FLOAT, {}, value, name);
        // We map to random vertex because they'll be accessed from everywhere
        graph.setTileMapping(tensors[name], rand() % graph.getTarget().getNumTiles());
    }
}

Program incrementTimestep(Graph &graph, std::map<std::string, Tensor> &tensors) {
    Sequence s;
    popops::addInPlace(graph, tensors["timestep"], tensors["1u"], s, "timestep++");
    return s;
}


void captureProfileInfo(Engine &engine)
{
    std::ofstream graphOfs;
    graphOfs.open("graph.json", std::ofstream::out | std::ofstream::trunc);

    std::ofstream executionOfs;
    executionOfs.open("execution.json", std::ofstream::out | std::ofstream::trunc);

    poplar::serializeToJSON(graphOfs, engine.getGraphProfile(), false);
    poplar::serializeToJSON(executionOfs, engine.getExecutionProfile(), false);

    graphOfs.close();
    executionOfs.close();
}

int main()
{
    auto device = getIpuModel();
    auto maxIters = 20;

    std::map<std::string, Tensor> tensors = {};

    poplar::Graph graph(device.getTarget());
    popops::addCodelets(graph);

    // # Lattice constants
    // v = np.array([ [1, 1], [1, 0],  [1,-1], [0,1], [0,0],
    //             [0,-1], [-1, 1], [-1, 0], [-1,-1]])
    // t = np.array([1/36, 1/9, 1/36, 1/9, 4/9, 1/9, 1/36, 1/9, 1/36])
    const std::vector<float> vInit = {1, 1, 1, 0, 1, -1, 0, 1, 0, 0, 0, -1, -1, 1, -1, 0, -1, -1};
    tensors["v"] = graph.addConstant(FLOAT, {9, 2}, vInit.data(), "v");
    poputil::mapTensorLinearly(graph, tensors["v"]);

    const std::vector<float> tInit = {1.0 / 36, 1.0 / 9, 1.0 / 36, 1.0 / 9, 4.0 / 9, 1.0 / 9, 1.0 / 36, 1.0 / 9, 1.0 / 36};
    tensors["t"] = graph.addConstant(FLOAT, {9}, tInit.data(), "t");
    poputil::mapTensorLinearly(graph, tensors["t"]);

    // u = np.zeros((2, nx, ny))
    tensors["u"] = graph.addVariable(FLOAT, {2, Constants.nx, Constants.ny}, "u");
    poputil::mapTensorLinearly(graph, tensors["u"]);

    // feq = np.zeros((9,nx,ny))
    for (auto name : {"fin", "feq", "fout"})
    {
        tensors[name] = graph.addVariable(FLOAT, {9, Constants.nx, Constants.ny}, name);
        poputil::mapTensorLinearly(graph, tensors[name]);
    }

    tensors["rho"] = graph.addVariable(FLOAT, {Constants.nx, Constants.ny}, "rho");
    poputil::mapTensorLinearly(graph, tensors["rho"]);

    tensors["1u"] = graph.addConstant(UNSIGNED_INT, {}, 1u, "1u");
        // We map to random vertex because they'll be accessed from everywhere
    graph.setTileMapping(tensors["1u"], rand() % graph.getTarget().getNumTiles());
    tensors["timestep"] = graph.addVariable(UNSIGNED_INT, {}, "timestep");
    graph.setTileMapping(tensors["timestep"], 5);
    graph.setInitialValue(tensors["timestep"], 0.0f);
    

    addScalarContants(
        graph, tensors,
        {
            {"0", 0.0f},
            {"0.5", 0.5f},
            {"1", 1.0f},
            {"2", 2.0f},
            {"3", 3.0f},
            {"3/2", 3.0 / 2.0f},
        });

    program::Sequence prog;

    auto equilibriumFn = graph.addFunction(equilibrium(graph, tensors));
    auto macroscopicFn = graph.addFunction(macroscopic(graph, tensors));
    auto incrementTimestepFn = graph.addFunction(incrementTimestep(graph, tensors));

    initialVelocity(graph, tensors);

    // fin = equilibrium(1, vel)
    {
        prog.add(Copy(tensors["vel"], tensors["u"]));
        // Set rho to 1 for first call
        Sequence s;
        popops::mul(graph, tensors["rho"], tensors["0"], s, "rho=0");
        popops::addInPlace(graph, tensors["rho"], tensors["1"], s, "rho+=1");
        prog.add(Call(equilibriumFn));
    }

    Program innerLoop =  Sequence();

    // for time in range(maxIter):
    prog.add(Repeat(maxIters, Sequence(innerLoop, Call(incrementTimestepFn))));

    auto engine = Engine(graph, prog, POPLAR_ENGINE_OPTIONS);

    engine.load(device);

    engine.run();

    // captureProfileInfo(*engine);
    // engine->printProfileSummary(std::cout,
    //                          OptionFlags{{"showExecutionSteps", "true"}});

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
