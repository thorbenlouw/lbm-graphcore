#include <poplar/Vertex.hpp>
#include <cstddef>
#include <print.h>
#include <math.h>
#include <ipudef.h>

/*
 * Slightly unintuitively we are using (0,0) as the bottom left corner
 */

using namespace poplar;

constexpr auto NumSpeeds = 9u;
enum SpeedIndexes {
    Middle, East, North, West, South, NorthEast, NorthWest, SouthWest, SouthEast
};

//#define DEBUG_CODELETS 0

/**
 * Convert the 9 directional speed distributions to a normed velocity
 */
class NormedVelocityVertex : public Vertex {
public:
    Input <Vector<float>> cells; // 9 speeds in every cell
    Output <Vector<float>> vels; // 1 velocity for every cell
    Input<unsigned> numCells;

    bool compute() {
        for (auto i = 0; i < *numCells; i++) {
            auto idx = i * NumSpeeds;
            const auto c0 = cells[idx + 0];
            const auto c1 = cells[idx + 1];
            const auto c2 = cells[idx + 2];
            const auto c3 = cells[idx + 3];
            const auto c4 = cells[idx + 4];
            const auto c5 = cells[idx + 5];
            const auto c6 = cells[idx + 6];
            const auto c7 = cells[idx + 7];
            const auto c8 = cells[idx + 8];


            auto local_density = (c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8);

            auto u_x = (c1 + c5 + c8 - c3 - c6 - c7) / local_density;
            auto u_y = (c2 + c5 + c6 - c4 - c7 - c8) / local_density;
            vels[i] = sqrtf((u_x * u_x) + (u_y * u_y));
        }
        return true;
    }
};


class NormedVelocityVertexOptim : public Vertex {
public:
    Input <Vector<float, VectorLayout::SCALED_PTR32, 4, false>> cells; // 9 speeds in every cell
    Output <Vector<float, VectorLayout::SCALED_PTR32, 4, false>> vels; // 1 velocity for every cell
    Input<unsigned> numCells;

    bool compute() {
        const uint16_t n = (uint16_t) * numCells;

        for (auto i = 0; i < n; i++) {
            auto cellAddress = reinterpret_cast<float2 *>(&cells[i * NumSpeeds]);

            auto c01 = *cellAddress;
            auto c23 = *(cellAddress + 1);
            auto c45 = *(cellAddress + 2);
            auto c67 = *(cellAddress + 3);
            auto c8X = *(cellAddress + 4);

            auto local_density_partial = c01 + c23 + c45 + c67 + float2{c8X[0], 0};
            auto local_density = local_density_partial[0] + local_density_partial[1];

            auto u_x = (c01[1] + c45[1] + c8X[0] - c23[1] - c67[0] - c67[1]) / local_density;
            auto u_y = (c23[0] + c45[1] + c67[0] - c45[0] - c67[1] - c8X[0]) / local_density;
            vels[i] = sqrtf((u_x * u_x) + (u_y * u_y));
        }
        return true;
    }
};

/**
 * For each velocity, if it is not masked, count it and add its velocity to the total
 */
class MaskedSumPartial : public Vertex { // On each worker, reduce the cells

public:
    Input <Vector<float>> velocities;
    Input <Vector<bool>> obstacles;
    Input<unsigned> numCells;
    Output <Vector<float>> totalAndCount;

    bool compute() {
        const uint16_t n = (uint16_t) * numCells;

        auto count = 0.0f;
        auto tmp = 0.0f;
        for (auto i = 0; i < n; i++) {
            tmp += velocities[i] * (1 - obstacles[i]); // only if not obstacle
            count += (1 - obstacles[i]); // only if not obstacle
        }
        totalAndCount[0] = tmp;
        totalAndCount[1] = count;

        return true;
    }
};

class MaskedSumPartialOptim : public Vertex { // On each worker, reduce the cells

public:
    Input <Vector<float, VectorLayout::ONE_PTR, 4, false>> velocities;
    Input <Vector<bool, VectorLayout::ONE_PTR, 4, false>> obstacles;
    Input<unsigned> numCells;
    Output <Vector<float, VectorLayout::SCALED_PTR32, 4, false>> totalAndCount;

    bool compute() {
        const uint16_t n = (uint16_t) * numCells;

        auto count = 0.0f;
        auto tmp = 0.0f;
        for (auto i = 0; i < n; i++) {
            tmp += velocities[i] * (1 - obstacles[i]); // only if not obstacle
            count += (1 - obstacles[i]); // only if not obstacle
        }
        float2 *f2out = reinterpret_cast<float2 *>(&totalAndCount[0]);
        f2out[0] = {tmp, count};
        return true;
    }
};


/**
 * Take a partial list of velocities and their counts and return one total velocity and total count
 */
class ReducePartials : public Vertex { // Take the partials within a tile and reduce them
public:
    Input <Vector<float>> totalAndCountPartials;
    Input<unsigned> numPartials;
    Output <Vector<float>> totalAndCount;

    bool compute() {
        float tmp = 0.f;
        float count = 0.f;

        for (int i = 0; i < *numPartials; i++) {
            tmp += totalAndCountPartials[i * 2];
            count += totalAndCountPartials[i * 2 + 1];
        }
        totalAndCount[0] = tmp;
        totalAndCount[1] = count;
        return true;
    }
};

class ReducePartialsOptim : public Vertex { // Take the partials within a tile and reduce them
public:
    Input <Vector<float, VectorLayout::SCALED_PTR32, 4>> totalAndCountPartials;
    Input<unsigned> numPartials;
    Output <Vector<float, VectorLayout::SCALED_PTR32, 4, false>> totalAndCount;

    bool compute() {
        float2 tmp = {0.0f, 0.0f};

        for (int i = 0; i < *numPartials; i++) {
            tmp += *reinterpret_cast<float2 *>(&totalAndCountPartials[i * 2]);
        }
        float2 *f2out = reinterpret_cast<float2 *>(&totalAndCount[0]);
        f2out[0] = tmp;
        return true;
    }
};

/**
 *  The output array of average velocities is spread throughout the distributed memories. We give each
 *  tile a vertex that knows what bits of the array are mapped to it. The totalAndCount is broadcast to
 *  each tile, and only the tile owning the memory writes it.
 */
class AppendReducedSum : public Vertex { // Reduce the per-tile partial sums and append to the average list

public:
    Input <Vector<float>> totalAndCount; // float2 of total|count
    Input<unsigned> indexToWrite;
    Input<unsigned> myStartIndex; // The index where my array starts
    Input<unsigned> myEndIndex; // My last index
    InOut <Vector<float>> finals; // The piece of the array I have

    bool compute() {

        const auto idx = *indexToWrite;
        if ((idx >= *myStartIndex) && (idx <= *myEndIndex)) {
            auto total = totalAndCount[0];
            auto count = totalAndCount[1];
            finals[idx - *myStartIndex] = total / count;
        }
        return true;
    }
};

class IncrementIndex : public Vertex {
public:
    InOut<unsigned> index;

    auto compute() -> bool {
        (*index)++;
        return true;
    }
};

class AccelerateFlowVertex : public Vertex {

public:
    InOut <Vector<float>> cellsInSecondRow; // 9 speeds in every cell, no halo
    Input <Vector<bool>> obstaclesInSecondRow;
    Input<unsigned> partitionWidth;
    Input<float> density;
    Input<float> accel;

    bool compute() {
        float w1 = *density * *accel / 9.f;
        float w2 = *density * *accel / 36.f;

        for (auto col = 0u; col < *partitionWidth; col++) {
            auto cellOffset = NumSpeeds * col;

            /* if the cell is not occupied and we don't send a negative density */
            if (!obstaclesInSecondRow[col]
                && (cellsInSecondRow[cellOffset + 3] - w1) > 0.f
                && (cellsInSecondRow[cellOffset + 6] - w2) > 0.f
                && (cellsInSecondRow[cellOffset + 7] - w2) > 0.f) {
                /* increase 'east-side' densities */
                cellsInSecondRow[cellOffset + 1] += w1;
                cellsInSecondRow[cellOffset + 5] += w2;
                cellsInSecondRow[cellOffset + 8] += w2;
                /* decrease 'west-side' densities */
                cellsInSecondRow[cellOffset + 3] -= w1;
                cellsInSecondRow[cellOffset + 6] -= w2;
                cellsInSecondRow[cellOffset + 7] -= w2;
            }
        }
        return true;
    }
};

class ReboundVertex : public Vertex {

public:
    Input <Vector<float>> in; // 9 speeds in every cell, no halo
    Output <Vector<float>> out; // 9 speeds in every cell, no halo
    Input <Vector<bool>> obstacles; //  no halo
    Input<unsigned> numRows; // no halo
    Input<unsigned> numCols; // no halo

    bool compute() {
        size_t cols = *numCols;
        size_t rows = *numRows;

        /* loop over the cells in the grid */
        for (size_t jj = 0; jj < rows; jj++) {
#pragma unroll 4
            for (size_t ii = 0; ii < cols; ii++) {
                auto obstacleIdx = jj * cols + ii;
                auto cellIdx = jj * (cols * NumSpeeds) + (ii * NumSpeeds);
                /* if the cell contains an obstacle */
                if (obstacles[obstacleIdx]) {
                    /* called after propagate, so taking values from scratch space
                    ** mirroring, and writing into main grid */
                    out[cellIdx + 1] = in[cellIdx + 3];
                    out[cellIdx + 2] = in[cellIdx + 4];
                    out[cellIdx + 3] = in[cellIdx + 1];
                    out[cellIdx + 4] = in[cellIdx + 2];
                    out[cellIdx + 5] = in[cellIdx + 7];
                    out[cellIdx + 6] = in[cellIdx + 8];
                    out[cellIdx + 7] = in[cellIdx + 5];
                    out[cellIdx + 8] = in[cellIdx + 6];
                }
            }
        }
        return true;
    }
};


class CollisionVertex : public Vertex {

public:
    Input <Vector<float>> in; // 9 speeds in every cell, no halo
    Input <Vector<bool>> obstacles; //  no halo
    Input<unsigned> numRows; // no halo
    Input<unsigned> numCols; // no halo
    Input<float> omega;
    Output <Vector<float>> out;

    bool compute() {
        const auto c_sq = 1.f / 3.f; /* square of speed of sound */
        const auto w0 = 4.f / 9.f;  /* weighting factor */
        const auto w1 = 1.f / 9.f;  /* weighting factor */
        const auto w2 = 1.f / 36.f; /* weighting factor */

        /* loop over the cells in the grid
        ** NB the collision step is called after
        ** the propagate step and so values of interest
        ** are in the scratch-space grid */
        for (int jj = 0; jj < *numRows; jj++) {
            for (int ii = 0; ii < *numCols; ii++) {
                const auto obstacleIdx = ii + jj * *numCols;
                const auto idx = obstacleIdx * NumSpeeds;
                /* don't consider occupied cells */
                // TODO can just fold rebound in here.
                if (!obstacles[obstacleIdx]) {
                    /* compute local density total */
                    auto local_density = 0.f;

                    for (int kk = 0; kk < NumSpeeds; kk++) {
                        local_density += in[idx + kk];
                    }

                    /* compute x velocity component */
                    const auto u_x = ((in[idx + 1] + in[idx + 5] + in[idx + 8])
                                      - (in[idx + 3] + in[idx + 6] + in[idx + 7]))
                                     / local_density;
                    /* compute y velocity component */
                    const auto u_y = ((in[idx + 2] + in[idx + 5] + in[idx + 6])
                                      - (in[idx + 4] + in[idx + 7] + in[idx + 8])) / local_density;

                    /* velocity squared */
                    const auto u_sq = u_x * u_x + u_y * u_y;

                    /* directional velocity components */
                    const float u[NumSpeeds] = {
                            0, /* middle */
                            u_x,        /* east */
                            u_y,  /* north */
                            -u_x,        /* west */
                            -u_y,  /* south */
                            u_x + u_y,  /* north-east */
                            -u_x + u_y,  /* north-west */
                            -u_x - u_y,  /* south-west */
                            u_x - u_y  /* south-east */
                    };


                    const auto u_over_2csq = u_sq / (2.f * c_sq);
                    const auto cc2 = (2.f * c_sq * c_sq);


                    // TODO! This is nicely vectorisable
                    /* equilibrium densities */
                    float d_equ[NumSpeeds];
                    /* zero velocity density: weight w0 */
                    d_equ[0] = w0 * local_density * (1.f - u_over_2csq);
                    /* axis speeds: weight w1 */
                    d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq + (u[1] * u[1]) / cc2 - u_over_2csq);
                    d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq + (u[2] * u[2]) / cc2 - u_over_2csq);
                    d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq + (u[3] * u[3]) / cc2 - u_over_2csq);
                    d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq + (u[4] * u[4]) / cc2 - u_over_2csq);
                    /* diagonal speeds: weight w2 */
                    d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq + (u[5] * u[5]) / cc2 - u_over_2csq);
                    d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq + (u[6] * u[6]) / cc2 - u_over_2csq);
                    d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq + (u[7] * u[7]) / cc2 - u_over_2csq);
                    d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq + (u[8] * u[8]) / cc2 - u_over_2csq);

                    /* relaxation step */
                    //TODO can float2 this
                    for (int kk = 0; kk < NumSpeeds; kk++) {
                        out[idx + kk] = in[idx + kk] + omega * (d_equ[kk] - in[idx + kk]);
                    }
                }
            }
        }
        return true;
    }
};


/* propagate densities from neighbouring cells, following
           ** appropriate directions of travel and writing into
           ** scratch space grid */
class PropagateVertex : public Vertex {

public:
    Input <Vector<float>> haloTop; // numCols x NumSpeeds
    Input <Vector<float>> haloLeft; // numRows x NumSpeeds
    Input <Vector<float>> haloRight; // numRows x NumSpeeds
    Input <Vector<float>> haloBottom; // numCols x NumSpeeds
    // We don't have vectors for these because we know exactly which speed cell we'll need:
    Input<float> haloTopLeft;
    Input<float> haloTopRight;
    Input<float> haloBottomLeft;
    Input<float> haloBottomRight;

    Input <Vector<float>> in; // numCols x numRows x NumSpeeds
    Input<unsigned> numRows; // i.e. excluding halo
    Input<unsigned> numCols; // i.e. excluding halo

    Output <Vector<float>> out; // numCols x numRows x NumSpeeds

    bool compute() {
        const auto nc = *numCols;
        const auto nr = *numRows;

        // Remember layout is (0,0) = bottom left
        const auto TOP = nr - 1;
        constexpr auto BOTTOM = 0ul;
        constexpr auto LEFT = 0ul;
        const auto RIGHT = nc - 1;
        const int northCellOffset = +((int) nc * NumSpeeds);
        const int southCellOffset = -((int) nc * NumSpeeds);
        constexpr int eastCellOffset = +(int) NumSpeeds;
        constexpr int westCellOffset = -(int) NumSpeeds;
        constexpr int middleCellOffset = 0;
        constexpr int sideHaloNorthCellOffset = +(int) NumSpeeds; // when we are using the left or right halo and want to know what up is
        constexpr int sideHaloSouthCellOffset = -(int) NumSpeeds;// when we are using the left or right halo and want to know what down is
        const int northWestCellOffset = +northCellOffset + westCellOffset;
        const int northEastCellOffset = +northCellOffset + eastCellOffset;
        const int southEastCellOffset = +southCellOffset + eastCellOffset;
        const int southWestCellOffset = +southCellOffset + westCellOffset;


        for (auto row = BOTTOM; row <= TOP; row++) {
            for (auto col = LEFT; col <= RIGHT; col++) {
                const int cellIdx = row * (nc * NumSpeeds) + col * NumSpeeds;
                // For each one we take the mirrored direction (so from the northCellSouthSpeed cell, we want the southCellNorthSpeed direction)
                const auto northCellSouthSpeed = [=]() -> float {
                    if (row == TOP) {
                        return haloTop[col * NumSpeeds + middleCellOffset + SpeedIndexes::South];
                    }
                    return in[cellIdx + northCellOffset + SpeedIndexes::South];
                };

                const auto northEastCellSouthWestSpeed = [=]() -> float {
                    if (row == TOP && col == RIGHT) {
                        return *haloTopRight;
                    } else if (row == TOP) {
                        return haloTop[col * NumSpeeds + eastCellOffset + SpeedIndexes::SouthWest];
                    } else if (col == RIGHT) {
                        return haloRight[row * NumSpeeds + sideHaloNorthCellOffset + SpeedIndexes::SouthWest];
                    }
                    return in[cellIdx + northEastCellOffset + SpeedIndexes::SouthWest];
                };

                const auto northWestCellSouthEastSpeed = [=]() -> float {
                    if (row == TOP && col == LEFT) {
                        return *haloTopLeft;
                    } else if (row == TOP) {
                        return haloTop[col * NumSpeeds + westCellOffset + SpeedIndexes::SouthEast];
                    } else if (col == LEFT) {
                        return haloLeft[row * NumSpeeds + sideHaloNorthCellOffset + SpeedIndexes::SouthEast];
                    }
                    return in[cellIdx + northWestCellOffset + SpeedIndexes::SouthEast];
                };

                const auto westCellEastSpeed = [=]() -> float {
                    if (col == LEFT) {
                        return haloLeft[row * NumSpeeds + middleCellOffset + SpeedIndexes::East];
                    }
                    return in[cellIdx + westCellOffset + SpeedIndexes::East];
                };

                const auto middle = [=]() -> float {
                    return in[cellIdx + middleCellOffset + SpeedIndexes::Middle];
                };

                const auto eastCellWestSpeed = [=]() -> float {
                    if (col == RIGHT) {
                        return haloRight[row * NumSpeeds + middleCellOffset + SpeedIndexes::West];
                    }
                    return in[cellIdx + eastCellOffset + SpeedIndexes::West];
                };

                const auto southCellNorthSpeed = [=]() -> float {
                    if (row == BOTTOM) {
                        return haloBottom[col * NumSpeeds + middleCellOffset + SpeedIndexes::North];
                    }
                    return in[cellIdx + southCellOffset + SpeedIndexes::North];
                };

                const auto southEastCellNorthWestSpeed = [=]() -> float {
                    if (row == BOTTOM && col == RIGHT) {
                        return *haloBottomRight;
                    } else if (row == BOTTOM) {
                        return haloBottom[col * NumSpeeds + eastCellOffset + SpeedIndexes::NorthWest];
                    } else if (col == RIGHT) {
                        return haloRight[row * NumSpeeds + sideHaloSouthCellOffset + SpeedIndexes::NorthWest];
                    }
                    return in[cellIdx + southEastCellOffset + SpeedIndexes::NorthWest];
                };


                const auto southWestCellNorthEastSpeed = [=]() -> float {
                    if (row == BOTTOM && col == LEFT) {
                        return *haloBottomLeft;
                    } else if (row == BOTTOM) {
                        return haloBottom[col * NumSpeeds + westCellOffset + SpeedIndexes::NorthEast];
                    } else if (col == LEFT) {
                        return haloLeft[row * NumSpeeds + sideHaloSouthCellOffset + SpeedIndexes::NorthEast];
                    }
                    return in[cellIdx + southWestCellOffset + SpeedIndexes::NorthEast];
                };
                // Now we copy the mirrored directions (South <-> North) etc.
                out[cellIdx + SpeedIndexes::Middle] = middle();
                out[cellIdx + SpeedIndexes::East] = westCellEastSpeed();
                out[cellIdx + SpeedIndexes::West] = eastCellWestSpeed();
                out[cellIdx + SpeedIndexes::South] = northCellSouthSpeed();
                out[cellIdx + SpeedIndexes::North] = southCellNorthSpeed();
                out[cellIdx + SpeedIndexes::NorthWest] = southEastCellNorthWestSpeed();
                out[cellIdx + SpeedIndexes::NorthEast] = southWestCellNorthEastSpeed();
                out[cellIdx + SpeedIndexes::SouthWest] = northEastCellSouthWestSpeed();
                out[cellIdx + SpeedIndexes::SouthEast] = northWestCellSouthEastSpeed();
            }
        }
        return true;
    }
};