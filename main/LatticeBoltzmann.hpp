//
// Created by Thorben Louw on 25/06/2020.
//

#ifndef LBM_GRAPHCORE_LATTICEBOLTZMANN_H
#define LBM_GRAPHCORE_LATTICEBOLTZMANN_H

#include <cmath>
#include <numeric>
#include "LbmParams.hpp"

namespace lbm {
    constexpr auto NumSpeeds = 9u;
    using Speeds  = float[NumSpeeds];

    enum SpeedIndexes {
        Middle, East, North, West, South, NorthEast, NorthWest, SouthWest, SouthEast
    };

    class Cells {
    private:
        const size_t nx;
        const size_t ny;
        std::unique_ptr<Speeds[]> data;

    public:


        explicit Cells(size_t nx, size_t ny) :
                nx(nx), ny(ny) {
            data = std::make_unique<Speeds[]>(nx * ny);
        }

        auto initialise(const Params &params) -> void {
            float w0 = params.density * 4.f / 9.f;
            float w1 = params.density / 9.f;
            float w2 = params.density / 36.f;

            for (auto jj = 0u; jj < ny; jj++) {
                for (auto ii = 0u; ii < nx; ii++) {
                    auto speeds = at(ii, jj);
                    speeds[0] = w0;
                    speeds[1] = w1;
                    speeds[2] = w1;
                    speeds[3] = w1;
                    speeds[4] = w1;
                    speeds[5] = w2;
                    speeds[6] = w2;
                    speeds[7] = w2;
                    speeds[8] = w2;
                }
            }
        }

        auto at(size_t x, size_t y) const -> float * {
            return data[x + nx * y];
        }

        auto averageVelocity(const Params &params, const Obstacles &obstacles) const -> float {
            int tot_cells = 0;  /* no. of cells used in calculation */
            float tot_u;          /* accumulated magnitudes of velocity for each cell */

            /* initialise */
            tot_u = 0.f;

            /* loop over all non-blocked cells */
            for (auto jj = 0u; jj < params.ny; jj++) {
                for (auto ii = 0u; ii < params.nx; ii++) {
                    int mask = 1 - obstacles.at(ii, jj);

                    auto speeds = at(ii, jj);
                    float local_density = std::accumulate<float[]>(speeds, speeds + NumSpeeds, 0.0f);

                    /* x-component of velocity */
                    float u_x = (float) mask * (speeds[1] + speeds[5] + speeds[8]
                                                - (speeds[3] + speeds[6] + speeds[7]))
                                / local_density;
                    /* compute y velocity component */
                    float u_y = (float) mask * ((speeds[2] + speeds[5] + speeds[6])
                                                - (speeds[4] + speeds[7] + speeds[8]))
                                / local_density;
                    /* accumulate the norm of x- and y- velocity components */
                    tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
                    /* increase counter of inspected cells */
                    tot_cells += mask;

                }
            }

            return tot_u / (float) tot_cells;
        }


        auto getData() -> float * {
            return (float *) data.get();
        };

        auto total_density() const -> float {
            auto tmp = (float *) (data.get());
            return std::accumulate<float[]>(tmp, tmp + nx * ny * NumSpeeds, 0.0f);
        }
    };


    auto reynoldsNumber(const Params &params, float average_velocity) -> float {
        const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);
        return average_velocity * params.reynolds_dim / viscosity;
    }

    auto writeAverageVelocities(const std::string &filename, const std::vector<float> &av_vels) -> bool {
        std::ofstream file;
        file.open(filename, std::ios::out);
        if (file.is_open()) {
            for (auto i = 0ul; i < av_vels.size(); i++) {
                file << i << ":\t" << std::scientific << std::setprecision(12)  << av_vels[i] << std::endl;
            }
            file.close();
            return true;
        }
        return false;
    }

    auto writeResults(const std::string &filename,
                      const Params &params,
                      const Obstacles &obstacles,
                      const Cells &cells) -> bool {
        const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
        float local_density;         /* per grid cell sum of densities */
        float pressure;              /* fluid pressure in grid cell */
        float u_x;                   /* x-component of velocity in grid cell */
        float u_y;                   /* y-component of velocity in grid cell */
        float u;                     /* norm--root of summed squares--of u_x and u_y */

        std::ofstream file;
        file.open(filename, std::ios::out);
        if (file.is_open()) {

            for (auto jj = 0u; jj < params.ny; jj++) {
                for (auto ii = 0u; ii < params.nx; ii++) {
                    /* an occupied cell */
                    if (obstacles.at(ii, jj)) {
                        u_x = u_y = u = 0.f;
                        pressure = params.density * c_sq;
                    }
                        /* no obstacle */
                    else {
                        local_density = 0.f;
                        auto speeds = cells.at(ii, jj);

                        for (auto kk = 0u; kk < NumSpeeds; kk++) {
                            local_density += speeds[kk];
                        }

                        /* compute x velocity component */
                        u_x = (speeds[1] + speeds[5] + speeds[8]
                               - (speeds[3] + speeds[6] + speeds[7]))
                              / local_density;
                        /* compute y velocity component */
                        u_y = (speeds[2] + speeds[5] + speeds[6]
                               - (speeds[4] + speeds[7] + speeds[8]))
                              / local_density;
                        /* compute norm of velocity */
                        u = sqrtf((u_x * u_x) + (u_y * u_y));
                        /* compute pressure */
                        pressure = local_density * c_sq;
                    }

                    /* write to file */
                    file << ii << " " << jj << " " << file.precision(12) << std::scientific
                         << u_x << " " << " " << u_y
                         << " " << u << " " << pressure << " "
                         << (int) obstacles.at(ii, jj)
                         << std::endl;
                }
            }
            file.close();
            return true;
        }
        return false;
    }


};


#endif //LBM_GRAPHCORE_LATTICEBOLTZMANN_H
