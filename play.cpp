#include <cstdlib>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ScaledAdd.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/CycleCount.hpp>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <chrono>
#include <poplar/Program.hpp>
#include <popops/Reduce.hpp>
#include <math.h>

using namespace poplar;
using namespace poplar::program;

poplar::Device getIpuModel()
{
    poplar::IPUModel ipuModel;
    ipuModel.numIPUs = 1;
    ipuModel.tilesPerIPU = 4;
    return ipuModel.createDevice();
}
//
//int main2()
//{
//    auto device = getIpuModel();
//
//    poplar::Graph graph(device.getTarget());
//    popops::addCodelets(graph);
//
//    program::Sequence prog;
//
//    auto a = graph.addVariable(poplar::FLOAT, {3, 3}, "a");
//    graph.createHostWrite("a-write", a);
//    poputil::mapTensorLinearly(graph, a);
//    auto aBuf = std::vector<float>{
//        1, 1, 1,
//        2, 2, 2,
//        3, 3, 3};
//
//    auto b = graph.addVariable(poplar::FLOAT, {3, 3}, "b");
//    poputil::mapTensorLinearly(graph, b);
//    graph.createHostWrite("b-write", b);
//    auto bBuf = std::vector<float>{
//        2, 2, 2,
//        3, 3, 3,
//        4, 4, 4};
//
//    auto c = graph.addVariable(poplar::FLOAT, {2, 3}, "c");
//    poputil::mapTensorLinearly(graph, c);
//    graph.createHostWrite("c-write", c);
//    auto cBuf = std::vector<float>{
//        2, 2, 2,
//        3, 3, 3};
//
//    auto TWO = graph.addConstant(poplar::FLOAT, {}, 2, "2");
//    poputil::mapTensorLinearly(graph, TWO);
//
//    auto ONE = graph.addConstant(poplar::FLOAT, {}, 1, "1");
//    poputil::mapTensorLinearly(graph, ONE);
//
//    auto PI = graph.addConstant(poplar::FLOAT, {}, (float)M_PI, "PI");
//    poputil::mapTensorLinearly(graph, PI);
//
//    prog.add(program::PrintTensor("a", a));
//    prog.add(program::PrintTensor("b", b));
//    prog.add(program::PrintTensor("a+b", popops::add(graph, a, b, prog, "a+b")));
//    prog.add(program::PrintTensor("a-b", popops::sub(graph, a, b, prog, "a-b")));
//    prog.add(program::PrintTensor("a/2", popops::div(graph, a, TWO, prog, "a/2"))); // Tensor  / scalar
//    prog.add(program::PrintTensor("a^2", popops::square(graph, a, prog, "a^2")));
//    prog.add(program::PrintTensor("a<b", popops::lt(graph, a, b, prog, "a < b")));
//    prog.add(program::PrintTensor("2-b", popops::sub(graph, TWO, b, prog, "2 - b"))); // 2 -b broadcasts? Yes!
//    prog.add(program::PrintTensor("a.T", a.transpose()));                             // 2 -b broadcasts? Yes!
//    prog.add(program::PrintTensor("a.roll(0,1)", a.dimRoll(0, 1)));
//    // prog.add(program::PrintTensor("a << 1",  popops::shiftLeft(graph, a, a, prog, "a << 1")));  // ???
//    popops::divInPlace(graph, a, TWO, prog, "a /= 2");
//    prog.add(program::PrintTensor("a/=2", a));
//    auto tmp = popops::reduce(graph, b, FLOAT, {0}, popops::ReduceParams(popops::Operation::ADD), prog);
//    prog.add(program::PrintTensor("sum(b, 0)", tmp));
//
//    //Copy between slices of a tensor (a[1,:] = b[1,:])
//    prog.add(program::PrintTensor("a before slice copy", a));
//    auto tmp1 = a.slice(1, 2, 0);
//    prog.add(program::PrintTensor("a[[1],:]", tmp1));
//    auto tmp2 = a.slice(2, 3, 0);
//    prog.add(program::PrintTensor("a[[2],:]", tmp2));
//    prog.add(Copy(tmp1, tmp2));
//    prog.add(program::PrintTensor("a after slice copy (a[3,:] = a[2,:])", a));
//
//    std::cout << "Size of slice a[0,:] (rank) " << a[0].rank() << std::endl;
//    std::cout << "Size of slice a[0,0] (rank) " << a[0][0].rank() << std::endl;
//    prog.add(program::PrintTensor("a.slice(2,3,1).slice(1,3,0)", a.slice(2, 3, 1).slice(1, 3, 0)));
//
//    // Things I will need:
//    //np.sum(fin, axis=0)
//    //np.zeros((2, nx, ny))
//    //  u[0,:,:] += v[i,0] * fin[i,:,:]
//    //  u /= rho
//    // roll = T.dimRoll(dimIdx, newIdx) NOT THE SAME THING. Don't think this is implemented
//    // transpose = T.transpose
//
//    auto altb = popops::lteq(graph, a, b, prog, "a <= sqr(a)");
//    tmp = popops::select(graph, a, b, altb, prog, "select");
//    prog.add(PrintTensor("a", a));
//    prog.add(PrintTensor("b", b));
//    prog.add(PrintTensor("a<=b", altb));
//    prog.add(PrintTensor("select(a,b,altb)", tmp));
//
//    prog.add(PrintTensor("slice", c.slice({1, 1}, {2, 3})));
//
//    auto engine = Engine(graph, prog);
//
//    engine.load(device);
//    engine.writeTensor("a-write", aBuf.data());
//    engine.writeTensor("b-write", bBuf.data());
//    engine.writeTensor("c-write", cBuf.data());
//
//    engine.run(); // trainProg
//
//    return EXIT_SUCCESS;
//}

auto captureProfileInfo(Engine &engine) {
    std::ofstream graphOfs;
    graphOfs.open("graph.json", std::ofstream::out | std::ofstream::trunc);

    std::ofstream executionOfs;
    executionOfs.open("execution.json", std::ofstream::out | std::ofstream::trunc);

    serializeToJSON(graphOfs, engine.getGraphProfile(), false);
    serializeToJSON(executionOfs, engine.getExecutionProfile(), false);

    graphOfs.close();
    executionOfs.close();
}

#ifdef DEBUG
const auto EngineOptions = OptionFlags{
        {"target.saveArchive",                "archive.a"},
        {"debug.instrument",                  "true"},
        {"debug.instrumentCompute",           "true"},
        {"debug.loweredVarDumpFile",          "vars.capnp"},
        {"debug.instrumentControlFlow",       "true"},
        {"debug.computeInstrumentationLevel", "tile"}};
#else
const auto EngineOptions = OptionFlags{};
#endif

int main(int argc, char *argv[])
{
    auto device = []() -> std::optional<Device> {
        DeviceManager manager = DeviceManager::createDeviceManager();

        // Attempt to connect to a single IPU
        for (auto &d : manager.getDevices(poplar::TargetType::IPU, 1)) {
            std::cerr << "Trying to attach to IPU " << d.getId();
            if (d.attach()) {
                std::cerr << " - attached" << std::endl;
                return std::optional<Device>{std::move(d)};
            } else {
                std::cerr << std::endl;
            }
        }
        std::cerr << "Error attaching to device" << std::endl;
        return std::nullopt;
    }();

    auto graph=  poplar::Graph(device->getTarget());
    popops::addCodelets(graph);

    auto hostBuffer = std::vector<float>{
            1, 1, 1,
            2, 2, 2,
            3, 3, 3};
    std::cout << "EHRE" << std::endl;
    auto remoteBuffer = graph.addHostToDeviceFIFO("buffy", FLOAT, 9);
    std::cout << "EHRE" << std::endl;

    auto a = graph.addVariable(poplar::FLOAT, {3000, 3000}, "a");
    std::cout << "EHRE" << std::endl;

    poputil::mapTensorLinearly(graph, a);
    auto b = graph.addVariable(poplar::FLOAT, {3000, 3000}, "b");
    poputil::mapTensorLinearly(graph, b);

    auto firstCopy = Sequence();
//    firstCopy.add(Copy(remoteBuffer, a));
    auto secondCopy = Sequence();
//    secondCopy.add(Copy(remoteBuffer, b));
    auto prog = Sequence();
    popops::addInPlace(graph, a, b, prog, "a+b");
    secondCopy.add(Repeat( 10000, prog  ));
    auto timing = poplar::cycleCount(graph, secondCopy, 0, "timer");
    graph.createHostRead("readTimer", timing, true);

    auto engine = Engine(graph, {firstCopy, secondCopy}, EngineOptions);
    engine.load(*device);

    std::cout << "EHRE" << std::endl;
//    engine.connectStream( "buffy",hostBuffer.data());



    engine.run(0); // This program ("firstCopy") just gives us a maker in the execution trace

    auto tic = std::chrono::high_resolution_clock::now();
    engine.run(1); // We time this program ("secondCopy") two ways
    auto toc = std::chrono::high_resolution_clock::now();
    auto hostTiming = std::chrono::duration_cast < std::chrono::duration < double >> (toc - tic).count();

    unsigned long ipuTiming;
    engine.readTensor("readTimer", &ipuTiming);

    auto clockFreq = (double) device->getTarget().getTileClockFrequency();
    std::cout << "IPU reports " << clockFreq << "Hz clock frequency" << std::endl;
    std::cout << "IPU timing is: " <<  ipuTiming / clockFreq /1e6 << "s" << std::endl;
    std::cout << "Host timing is: " << hostTiming  / 1e6 <<  "s" <<std::endl ;

    return EXIT_SUCCESS;
}