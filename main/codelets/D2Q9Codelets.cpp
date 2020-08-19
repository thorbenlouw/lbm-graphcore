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

/**
 *  The output array of average velocities is spread throughout the distributed memories. We give each
 *  tile a vertex that knows what bits of the array are mapped to it. The totalAndCount is broadcast to
 *  each tile, and only the tile owning the memory writes it.
 */
class AppendReducedSum : public Vertex { // Reduce the per-tile partial sums and append to the average list

public:
    Input<float> total; // float2 of total|count
    Input<int> count; // float2 of total|count
    Input<unsigned> indexToWrite;
    Input<unsigned> myStartIndex; // The index where my array starts
    Input<unsigned> myEndIndex; // My last index
    InOut <Vector<float>> finals; // The piece of the array I have

    bool compute() {

        const auto idx = *indexToWrite;
        if ((idx >= *myStartIndex) && (idx <= *myEndIndex)) {
            finals[idx - *myStartIndex] = *total / *count;
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

// TODO: Idea change cells to SOA so that we can do everything vectorised (in 2s)
class CollisionVertex : public Vertex {

public:
    Input <Vector<float>> in; // 9 speeds in every cell, no halo
    Input <Vector<bool>> obstacles; //  no halo
    Input<unsigned> numRows; // no halo
    Input<unsigned> numCols; // no halo
    Input<float> omega;
    Output <Vector<float>> out;
    Output<float> normedVelocityPartial; // sum of normed velocities for non-obstacle cells
    Output<int> countPartial; // count of how many non-obstacle cells there are


    bool compute() {
        auto tmp_count = 0u;
        auto tmp_velocityPartial = 0.f;

        const auto c_sq = 1.f / 3.f; /* square of speed of sound */
        const auto cc2 = (2.f * c_sq * c_sq);
        const auto w0 = 4.f / 9.f;  /* weighting factor */
        const auto w1 = 1.f / 9.f;  /* weighting factor */
        const auto w2 = 1.f / 36.f; /* weighting factor */

        /* loop over the cells in the grid
        ** NB the collision step is called after
        ** the propagate step and so values of interest
        ** are in the scratch-space grid */
        const auto nr = *numRows;
        const auto nc = *numCols;
        const auto o = *omega;

        for (int jj = 0; jj < nr; jj++) {
            for (int ii = 0; ii < nc; ii++) {
                const auto idx = ii + jj * *numCols;
                const auto cellsIdx = idx * NumSpeeds;
                const auto rebound = obstacles[idx];
                if (rebound) {
                    out[cellsIdx + 1] = in[cellsIdx + 3];
                    out[cellsIdx + 2] = in[cellsIdx + 4];
                    out[cellsIdx + 3] = in[cellsIdx + 1];
                    out[cellsIdx + 4] = in[cellsIdx + 2];
                    out[cellsIdx + 5] = in[cellsIdx + 7];
                    out[cellsIdx + 6] = in[cellsIdx + 8];
                    out[cellsIdx + 7] = in[cellsIdx + 5];
                    out[cellsIdx + 8] = in[cellsIdx + 6];
                } else {
                    /* compute local density total */
                    auto local_density = 0.f;

                    for (int kk = 0; kk < NumSpeeds; kk++) {
                        local_density += in[cellsIdx + kk];
                    }

                    /* compute x velocity component */
                    const auto u_x = ((in[cellsIdx + 1] + in[cellsIdx + 5] + in[cellsIdx + 8])
                                      - (in[cellsIdx + 3] + in[cellsIdx + 6] + in[cellsIdx + 7]))
                                     / local_density;
                    /* compute y velocity component */
                    const auto u_y = ((in[cellsIdx + 2] + in[cellsIdx + 5] + in[cellsIdx + 6])
                                      - (in[cellsIdx + 4] + in[cellsIdx + 7] + in[cellsIdx + 8])) / local_density;

                    /* velocity squared */
                    const auto u_sq = u_x * u_x + u_y * u_y;
                    tmp_velocityPartial += sqrtf(u_sq);
                    tmp_count++;

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
                        out[cellsIdx + kk] = in[cellsIdx + kk] + o * (d_equ[kk] - in[cellsIdx + kk]);
                    }
                }
            }
        }
        *countPartial = tmp_count;
        *normedVelocityPartial = tmp_velocityPartial;
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