

#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <random>
#include <cxxopts.hpp>

#include "include/LbmParams.hpp"
#include "include/LatticeBoltzmannUtils.hpp"


auto
accelerate_flow(const lbm::Params &params, const std::unique_ptr<float[]> &cells,
                const std::unique_ptr<bool[]> &obstacles) -> void {
    const auto at = [&params](auto row, auto col) -> size_t { return row * params.nx + col; };
    const auto row = params.ny - 2;

    const auto w1 = params.density * params.accel / 9.f;
    const auto w2 = params.density * params.accel / 36.f;

    for (auto col = 0u; col < params.nx; col++) {

    }
}

auto
rebound(const lbm::Params &params, const std::unique_ptr<float[]> &cells,
        const std::unique_ptr<bool[]> &obstacles) -> void {

}

auto
stream(const lbm::Params &params, const std::unique_ptr<float[]> &cells,
       const std::unique_ptr<bool[]> &obstacles) -> void {

}

auto
collide(const lbm::Params &params, const std::unique_ptr<float[]> &cells,
        const std::unique_ptr<bool[]> &obstacles) -> void {

}

auto averageVelocity(const lbm::Params &params, const std::unique_ptr<float[]> &cells,
                     const std::unique_ptr<bool[]> &obstacles) -> float {
    return 0.f;
}

auto timestep(const lbm::Params &params, const std::unique_ptr<float[]> &cells,
              const std::unique_ptr<bool[]> &obstacles) -> void {
    accelerate_flow(params, cells, obstacles);
    rebound(params, cells, obstacles);
    stream(params, cells, obstacles);
    collide(params, cells, obstacles);
}

auto main(int argc, char *argv[]) -> int {
    std::string outputFilename, paramsFileArg, obstaclesFileArg;

    cxxopts::Options options(argv[0], "D2Q9 BGK Lattice Boltzmann on 1 CPU core");
    options.add_options()
            ("params", "filename of parameters file", cxxopts::value<std::string>(paramsFileArg))
            ("obstacles", "filename of obstacles file", cxxopts::value<std::string>(obstaclesFileArg));
    auto opts = options.parse(argc, argv);

    auto params = lbm::Params::fromFile(paramsFileArg);
    if (!params.has_value()) {
        std::cerr << "Could not parse parameters file. Aborting" << std::endl;
        return EXIT_FAILURE;
    }


    auto obstacles = lbm::Obstacles::fromFile(params->nx, params->ny, obstaclesFileArg);
    if (!obstacles.has_value()) {
        std::cerr << "Could not parse obstacles file" << std::endl;
        return EXIT_FAILURE;
    }


    auto av_vel = std::unique_ptr<float[]>{new float[params->maxIters]};
    auto cells = std::unique_ptr<float[]>{new float[params->ny * params->nx * lbm::NumSpeeds]};

    for (auto iter = 0u; iter < params->maxIters; iter++) {
        timestep(*params, cells, obstacles->data_ptr());
        av_vel[iter] = averageVelocity(*params, cells, obstacles->data_ptr());
    }

    std::cout << "==done==" << std::endl;

    return EXIT_SUCCESS;
}
