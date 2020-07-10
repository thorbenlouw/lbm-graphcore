//
// Created by Thorben Louw on 25/06/2020.
//
#ifndef LBM_GRAPHCORE_GRAPHCOREUTILS_H
#define LBM_GRAPHCORE_GRAPHCOREUTILS_H

#include <cstdlib>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/Program.hpp>

#include <poplar/DeviceManager.hpp>
#include <iomanip>
#include <iostream>
#include <fstream>

using namespace poplar;
using namespace poplar::program;

namespace lbm {

    typedef  std::map<std::string, Tensor> TensorMap;


    const auto POPLAR_ENGINE_OPTIONS_DEBUG = OptionFlags{
            {"target.saveArchive",                "archive.a"},
            {"debug.instrument",                  "true"},
            {"debug.instrumentCompute",           "true"},
            {"debug.loweredVarDumpFile",          "vars.capnp"},
            {"debug.instrumentControlFlow",       "true"},
            {"debug.computeInstrumentationLevel", "tile"}};

    const auto POPLAR_ENGINE_OPTIONS_NODEBUG = OptionFlags{};

    auto getIpuModel(const unsigned short numIpus=1) -> std::optional<Device> {
        IPUModel ipuModel;
        ipuModel.numIPUs = numIpus;
        ipuModel.tilesPerIPU = 1216;
        return {ipuModel.createDevice()};
    }

    auto captureProfileInfo(Engine &engine, Graph &graph) {
        std::ofstream graphOfs;
        graphOfs.open("graph.json", std::ofstream::out | std::ofstream::trunc);

        std::ofstream executionOfs;
        executionOfs.open("execution.json", std::ofstream::out | std::ofstream::trunc);

        serializeToJSON(graphOfs, engine.getGraphProfile(), false);
        serializeToJSON(executionOfs, engine.getExecutionProfile(), false);
        
        graphOfs.close();
        executionOfs.close();


        std::ofstream graphSerOfs;
        graphSerOfs.open("serialized_graph.capnp", std::ofstream::out | std::ofstream::trunc);

        graph.serialize(graphSerOfs, poplar::SerializationFormat::Binary);
        graphSerOfs.close();
    }

    auto getIpuDevice(unsigned int numIpus = 1) -> std::optional<Device> {
        DeviceManager manager = DeviceManager::createDeviceManager();

        // Attempt to connect to a single IPU
        for (auto &d : manager.getDevices(poplar::TargetType::IPU, numIpus)) {
            std::cerr << "Trying to attach to IPU " << d.getId();
            if (d.attach()) {
                std::cerr << " - attached" << std::endl;
                return {std::move(d)};
            } else {
                std::cerr << std::endl;
            }
        }
        std::cerr << "Error attaching to device" << std::endl;
        return std::nullopt;
    }

    auto createDebugEngine(Graph &graph, ArrayRef<Program> programs) -> Engine {
        return Engine(graph, programs, POPLAR_ENGINE_OPTIONS_DEBUG);
    }


    auto createReleaseEngine(Graph &graph, ArrayRef<Program> programs) -> Engine {
        return Engine(graph, programs, POPLAR_ENGINE_OPTIONS_NODEBUG);
    }
}

#endif //LBM_GRAPHCORE_GRAPHCOREUTILS_H
