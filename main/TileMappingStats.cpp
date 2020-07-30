
#include <cstdlib>
#include <cxxopts.hpp>
#include "StructuredGridUtils.hpp"
#include <chrono>
#include <iostream>
#include <sstream>

int main(int argc, char *argv[]) {
    std::string outputFilename;
    unsigned numIpus = 1u;
    unsigned minWidth = 128u;
    unsigned minHeight = 128u;
    unsigned maxWidth = 4000u;
    unsigned maxHeight = 4000u;
    unsigned numSamples = 10000u;

    cxxopts::Options options(argv[0],
                             " - Work out the stats for a range of matrix sizes after they have been mapped");
    options.add_options()
            ("num-ipus", "Number of IPUs to target (1,2,4,8 or 16)", cxxopts::value<unsigned>(numIpus))
            ("min-width", "Min width", cxxopts::value<unsigned>(minWidth))
            ("max-width", "Max width", cxxopts::value<unsigned>(maxWidth))
            ("min-height", "Min height", cxxopts::value<unsigned>(minHeight))
            ("n,num-samples", "Number of samples", cxxopts::value<unsigned>(numSamples)->default_value("100000"))
            ("max-height", "Max height", cxxopts::value<unsigned>(maxHeight));
    try {
        auto opts = options.parse(argc, argv);
        if (opts.count("min-width") + opts.count("min-height") + opts.count("max-width") + opts.count("max-height") <
            4) {
            std::cerr << options.help() << std::endl;
            return EXIT_FAILURE;
        }
    } catch (cxxopts::OptionParseException &) {
        std::cerr << options.help() << std::endl;
        return EXIT_FAILURE;
    }


    constexpr auto NumTiles = 1216;
    constexpr auto NumWorkers = 6;


    std::cout << "numIpus,width,height,wastedTiles,wastedWorkers,loadBalance,maxSpeedup" << std::endl;


//    for (auto sample = 0u; sample < numSamples; sample++) {
//        auto height = rand() % (maxHeight - minHeight + 1) + minHeight;
//        auto width = rand() % (maxWidth - minWidth + 1) + minWidth;
//        auto ipuLevelMappings = grids::partitionForIpus(
//                {height, width}, numIpus,
//                (unsigned) std::min(4000 * 4000.f, (maxHeight * maxWidth) / (float) numIpus));
//        if (!ipuLevelMappings.has_value()) { // we can't fit this size onto the IPU
//            break;
//        }
//        auto tileLevelMappings = grids::toTilePartitions(*ipuLevelMappings, NumTiles);
//        auto workerLevelMappings = grids::toWorkerPartitions(tileLevelMappings);
//
//        // what size of the input area a '100% busy worker' will be
//        auto busiestWorker = 0u;
//        for (const auto &[target, slice]: workerLevelMappings) {
//            const auto busyness = slice.width() * slice.height();
//            if (busyness > busiestWorker) busiestWorker = busyness;
//        }
//
//        double loadBalanceTotal = 0.0;
//        double wasteWorkers = NumTiles * numIpus * NumWorkers - workerLevelMappings.size();
//        double wasteTiles = NumTiles * numIpus - tileLevelMappings.size();
//        for (const auto &[target, slice]: workerLevelMappings) {
//            auto workerLoad = slice.width() * slice.height();
//            loadBalanceTotal += workerLoad;
//        }
//        double aveLoadBalance = (loadBalanceTotal / workerLevelMappings.size()) / busiestWorker * 100;
//        double maxSpeedup = ((double) width * height) / ((double) busiestWorker);
//        std::cout << numIpus << "," << width << "," << height << "," << wasteTiles << "," << wasteWorkers << ","
//                  << aveLoadBalance << "," << maxSpeedup << std::endl;
//
//
//    }


    for (double height = minHeight; height <= maxHeight; height += ((maxHeight - minHeight) / 200.0)) {
        for (double width = minWidth; width <= maxWidth; width += ((maxWidth - minWidth) / 200.0)) {
            auto ipuLevelMappings = grids::partitionForIpus(
                    {(unsigned) height, (unsigned) width}, numIpus,
                    (unsigned) std::min(4000 * 4000.f, (maxHeight * maxWidth) / (float) numIpus));
            if (!ipuLevelMappings.has_value()) { // we can't fit this size onto the IPU
                break;
            }
            auto tileLevelMappings = grids::toTilePartitions(*ipuLevelMappings, NumTiles);
            auto workerLevelMappings = grids::toWorkerPartitions(tileLevelMappings);

            // what size of the input area a '100% busy worker' will be
            auto busiestWorker = 0u;
            for (const auto &[target, slice]: workerLevelMappings) {
                const auto busyness = slice.width() * slice.height();
                if (busyness > busiestWorker) busiestWorker = busyness;
            }

            double loadBalanceTotal = 0.0;
            double wasteWorkers = NumTiles * numIpus * NumWorkers - workerLevelMappings.size();
            double wasteTiles = NumTiles * numIpus - tileLevelMappings.size();
            for (const auto &[target, slice]: workerLevelMappings) {
                auto workerLoad = slice.width() * slice.height();
                loadBalanceTotal += workerLoad;
            }
            double aveLoadBalance = (loadBalanceTotal / workerLevelMappings.size()) / busiestWorker * 100;
            double maxSpeedup = ((double) width * height) / ((double) busiestWorker);
            std::cout << numIpus << "," << (int) width << "," << (int) height << "," << wasteTiles << "," << wasteWorkers << ","
                      << aveLoadBalance << "," << maxSpeedup << std::endl;

        }
    }


    return EXIT_SUCCESS;
}