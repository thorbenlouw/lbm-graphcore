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

int main()
{
    auto device = getIpuModel();

    poplar::Graph graph(device.getTarget());
    popops::addCodelets(graph);

    program::Sequence prog;

    auto a = graph.addVariable(poplar::FLOAT, {3, 3}, "a");
    graph.createHostWrite("a-write", a);
    poputil::mapTensorLinearly(graph, a);
    auto aBuf = std::vector<float>{
        1, 1, 1,
        2, 2, 2,
        3, 3, 3};

    auto b = graph.addVariable(poplar::FLOAT, {3, 3}, "b");
    poputil::mapTensorLinearly(graph, b);
    graph.createHostWrite("b-write", b);
    auto bBuf = std::vector<float>{
        2, 2, 2,
        3, 3, 3,
        4, 4, 4};

    auto c = graph.addVariable(poplar::FLOAT, {2, 3}, "c");
    poputil::mapTensorLinearly(graph, c);
    graph.createHostWrite("c-write", c);
    auto cBuf = std::vector<float>{
        2, 2, 2,
        3, 3, 3};

    auto TWO = graph.addConstant(poplar::FLOAT, {}, 2, "2");
    poputil::mapTensorLinearly(graph, TWO);

    auto ONE = graph.addConstant(poplar::FLOAT, {}, 1, "1");
    poputil::mapTensorLinearly(graph, ONE);

    auto PI = graph.addConstant(poplar::FLOAT, {}, (float)M_PI, "PI");
    poputil::mapTensorLinearly(graph, PI);

    prog.add(program::PrintTensor("a", a));
    prog.add(program::PrintTensor("b", b));
    prog.add(program::PrintTensor("a+b", popops::add(graph, a, b, prog, "a+b")));
    prog.add(program::PrintTensor("a-b", popops::sub(graph, a, b, prog, "a-b")));
    prog.add(program::PrintTensor("a/2", popops::div(graph, a, TWO, prog, "a/2"))); // Tensor  / scalar
    prog.add(program::PrintTensor("a^2", popops::square(graph, a, prog, "a^2")));
    prog.add(program::PrintTensor("a<b", popops::lt(graph, a, b, prog, "a < b")));
    prog.add(program::PrintTensor("2-b", popops::sub(graph, TWO, b, prog, "2 - b"))); // 2 -b broadcasts? Yes!
    prog.add(program::PrintTensor("a.T", a.transpose()));                             // 2 -b broadcasts? Yes!
    prog.add(program::PrintTensor("a.roll(0,1)", a.dimRoll(0, 1)));
    // prog.add(program::PrintTensor("a << 1",  popops::shiftLeft(graph, a, a, prog, "a << 1")));  // ???
    popops::divInPlace(graph, a, TWO, prog, "a /= 2");
    prog.add(program::PrintTensor("a/=2", a));
    auto tmp = popops::reduce(graph, b, FLOAT, {0}, popops::ReduceParams(popops::Operation::ADD), prog);
    prog.add(program::PrintTensor("sum(b, 0)", tmp));

    //Copy between slices of a tensor (a[1,:] = b[1,:])
    prog.add(program::PrintTensor("a before slice copy", a));
    auto tmp1 = a.slice(1, 2, 0);
    prog.add(program::PrintTensor("a[[1],:]", tmp1));
    auto tmp2 = a.slice(2, 3, 0);
    prog.add(program::PrintTensor("a[[2],:]", tmp2));
    prog.add(Copy(tmp1, tmp2));
    prog.add(program::PrintTensor("a after slice copy (a[3,:] = a[2,:])", a));

    std::cout << "Size of slice a[0,:] (rank) " << a[0].rank() << std::endl;
    std::cout << "Size of slice a[0,0] (rank) " << a[0][0].rank() << std::endl;
    prog.add(program::PrintTensor("a.slice(2,3,1).slice(1,3,0)", a.slice(2, 3, 1).slice(1, 3, 0)));

    // Things I will need:
    //np.sum(fin, axis=0)
    //np.zeros((2, nx, ny))
    //  u[0,:,:] += v[i,0] * fin[i,:,:]
    //  u /= rho
    // roll = T.dimRoll(dimIdx, newIdx) NOT THE SAME THING. Don't think this is implemented
    // transpose = T.transpose

    auto altb = popops::lteq(graph, a, b, prog, "a <= sqr(a)");
    tmp = popops::select(graph, a, b, altb, prog, "select");
    prog.add(PrintTensor("a", a));
    prog.add(PrintTensor("b", b));
    prog.add(PrintTensor("a<=b", altb));
    prog.add(PrintTensor("select(a,b,altb)", tmp));

    prog.add(PrintTensor("slice", c.slice({1, 1}, {2, 3})));

    auto engine = Engine(graph, prog);

    engine.load(device);
    engine.writeTensor("a-write", aBuf.data());
    engine.writeTensor("b-write", bBuf.data());
    engine.writeTensor("c-write", cBuf.data());

    engine.run(); // trainProg

    return EXIT_SUCCESS;
}