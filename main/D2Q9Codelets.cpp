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
 * Convert the 9 directional speed distributions to a normed velocity
 */
class NormedVelocityVertex : public Vertex {
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
 * Take a partial list of velocities and their counts, determine the  total velocity and total count, and then
 * record the average (total/count) at the given index in the array
 */
class AppendReducedSum : public Vertex { // Reduce the per-tile partial sums and append to the average list

public:
    Input <Vector<float, VectorLayout::SCALED_PTR64, 8>> totalAndCountPartials;
    Input<unsigned> index;
    Input<unsigned> numPartials;
    Output <Vector<float, VectorLayout::SCALED_PTR32, 4>> finals;

    bool compute() {

        float2 tmp = {0.0f, 0.0f};
        for (auto i = 0u; i < *numPartials; i++) {
            tmp += *reinterpret_cast<float2 *>(&totalAndCountPartials[i * 2]);
        }
        finals[*index] = tmp[0] / tmp[1];

        return true;
    }
};

class AccelerateFlowVertex : public Vertex {

public:
    Input <Vector<float>> cellsInSecondRow; // 9 speeds in every cell, no halo
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

    Input <Vector<float, VectorLayout::SCALED_PTR32, 4, false>> in; // numCols x numRows x NumSpeeds
    Input<unsigned> numRows; // i.e. excluding halo
    Input<unsigned> numCols; // i.e. excluding halo
    Input <Vector<float, VectorLayout::SCALED_PTR32, 4, false>> out; // numCols x numRows x NumSpeeds

    bool compute() {
        if (numRows < 2 || numCols < 2) {
            return false;
        }

        auto assignToCell = [this](const size_t idx, const float2 src[5]) -> void {
            float2 *f2out = reinterpret_cast<float2 *>(&out[idx]);
            f2out[0] = src[0];
            f2out[1] = src[1];
            f2out[2] = src[2];
            f2out[3] = src[3];
            out[idx + 8] = src[4][0];
        };

        // Remember layout is (0,0) = bottom left

        const int northCellOffset = +((int) numCols * NumSpeeds);
        const int southCellOffset = -((int) numCols * NumSpeeds);
        constexpr int eastCellOffset = +(int) NumSpeeds;
        constexpr int westCellOffset = -(int) NumSpeeds;
        constexpr int middleCellOffset = 0;
        constexpr int sideHaloNorthCellOffset = +(int) NumSpeeds; // when we are using the left or right halo and want to know what up is
        constexpr int sideHaloSouthCellOffset = -(int) NumSpeeds;// when we are using the left or right halo and want to know what down is
        const int northWestCellOffset = +northCellOffset - (int) NumSpeeds;
        const int northEastCellOffset = +northCellOffset + (int) NumSpeeds;
        const int southEastCellOffset = +southCellOffset + (int) NumSpeeds;
        const int southWestCellOffset = +southCellOffset - (int) NumSpeeds;

        // Top left - remember layout is (0,0) is bottom left
        auto topLeft = [=]() -> void {
            const auto row = (numRows - 1u);
            constexpr auto col = 0u;
            const int cellIdx = row * (numCols * NumSpeeds) + col * NumSpeeds;
            const auto n = haloTop[col * NumSpeeds + middleCellOffset + SpeedIndexes::North];
            const auto ne = haloTop[col * NumSpeeds + eastCellOffset + SpeedIndexes::NorthEast];
            const auto nw = *haloTopLeft;
            const auto w = haloLeft[row * NumSpeeds + middleCellOffset + SpeedIndexes::West];
            const auto m = in[cellIdx + middleCellOffset + SpeedIndexes::Middle];
            const auto e = in[cellIdx + eastCellOffset + SpeedIndexes::East];
            const auto s = in[cellIdx + southCellOffset + SpeedIndexes::South];
            const auto se = in[cellIdx + southEastCellOffset + SpeedIndexes::SouthEast];
            const auto sw = haloLeft[row * NumSpeeds + sideHaloSouthCellOffset + SpeedIndexes::SouthWest];

            float2 result[5] = {{m,  e},
                                {n,  w},
                                {s,  ne},
                                {nw, sw},
                                {se, 0.0f}};
            assignToCell(cellIdx, result);
        };

        // Top Right
        auto topRight = [=]() -> void {
            const auto row = (numRows - 1u);
            const auto col = (numCols - 1u);
            const int cellIdx = row * (numCols * NumSpeeds) + col * NumSpeeds;
            const auto n = haloTop[col * NumSpeeds + middleCellOffset + SpeedIndexes::North];
            const auto ne = *haloTopRight;
            const auto nw = haloTop[col * NumSpeeds + westCellOffset + SpeedIndexes::NorthWest];
            const auto w = in[cellIdx + westCellOffset + SpeedIndexes::West];
            const auto m = in[cellIdx + middleCellOffset + SpeedIndexes::Middle];
            const auto e = haloRight[row * NumSpeeds + middleCellOffset + SpeedIndexes::East];
            const auto s = in[cellIdx + southCellOffset + SpeedIndexes::South];
            const auto se = haloRight[row * NumSpeeds + sideHaloSouthCellOffset + SpeedIndexes::SouthEast];
            const auto sw = in[cellIdx + southWestCellOffset + SpeedIndexes::SouthWest];

            float2 result[5] = {{m,  e},
                                {n,  w},
                                {s,  ne},
                                {nw, sw},
                                {se, 0.0f}};
            assignToCell(cellIdx, result);
        };

        // Bottom Left
        auto bottomLeft = [=]() -> void {
            constexpr auto row = 0u;
            constexpr auto col = 0u;
            const auto cellIdx = row * (numCols * NumSpeeds) + col * NumSpeeds;
            const auto n = in[cellIdx + northCellOffset + SpeedIndexes::North];
            const auto ne = in[cellIdx + northEastCellOffset + SpeedIndexes::NorthEast];
            const auto nw = haloLeft[row * NumSpeeds + sideHaloNorthCellOffset + SpeedIndexes::NorthWest];
            const auto w = haloLeft[row * NumSpeeds + middleCellOffset + SpeedIndexes::West];
            const auto m = in[cellIdx + middleCellOffset + SpeedIndexes::Middle];
            const auto e = in[cellIdx + eastCellOffset + SpeedIndexes::East];
            const auto s = haloBottom[col * NumSpeeds + middleCellOffset + SpeedIndexes::South];
            const auto se = haloBottom[col * NumSpeeds + eastCellOffset + SpeedIndexes::SouthEast];
            const auto sw = haloBottomLeft;

            float2 result[5] = {{m,  e},
                                {n,  w},
                                {s,  ne},
                                {nw, sw},
                                {se, 0.0f}};
            assignToCell(cellIdx, result);
        };

        // Bottom right
        auto bottomRight = [=]() -> void {
            constexpr auto row = 0u;
            const auto col = (numCols - 1u);
            const auto cellIdx = row * (numCols * NumSpeeds) + col * NumSpeeds;
            const auto n = in[cellIdx + northCellOffset + SpeedIndexes::North];
            const auto ne = haloRight[row * NumSpeeds + sideHaloNorthCellOffset + SpeedIndexes::NorthEast];
            const auto nw = in[cellIdx + northWestCellOffset + SpeedIndexes::NorthWest];
            const auto w = in[cellIdx + westCellOffset + SpeedIndexes::West];
            const auto m = in[cellIdx + middleCellOffset + SpeedIndexes::Middle];
            const auto e = haloRight[row * NumSpeeds + middleCellOffset + SpeedIndexes::East];
            const auto s = haloBottom[col * NumSpeeds + middleCellOffset + SpeedIndexes::South];
            const auto se = *haloBottomRight;
            const auto sw = haloBottom[col * NumSpeeds + westCellOffset + SpeedIndexes::SouthWest];

            float2 result[5] = {{m,  e},
                                {n,  w},
                                {s,  ne},
                                {nw, sw},
                                {se, 0.0f}};
            assignToCell(cellIdx, result);
        };

        // Top
        auto top = [=]() -> void {
            auto row = (numRows - 1u);
            for (size_t col = 1; col < *numCols - 1; col++) {
                const int cellIdx = row * (numCols * NumSpeeds) + col * NumSpeeds;
                const auto n = haloTop[col * NumSpeeds + middleCellOffset + SpeedIndexes::North];
                const auto ne = haloTop[col * NumSpeeds + eastCellOffset + SpeedIndexes::NorthEast];
                const auto nw = haloTop[col * NumSpeeds + westCellOffset + SpeedIndexes::NorthWest];
                const auto w = in[cellIdx + westCellOffset + SpeedIndexes::West];
                const auto m = in[cellIdx + middleCellOffset + SpeedIndexes::Middle];
                const auto e = in[cellIdx + eastCellOffset + SpeedIndexes::East];
                const auto s = in[cellIdx + southCellOffset + SpeedIndexes::South];
                const auto se = in[cellIdx + southEastCellOffset + SpeedIndexes::SouthEast];
                const auto sw = in[cellIdx + southWestCellOffset + SpeedIndexes::SouthWest];

                float2 result[5] = {{m,  e},
                                    {n,  w},
                                    {s,  ne},
                                    {nw, sw},
                                    {se, 0.0f}};
                assignToCell(cellIdx, result);
            }
        };

        // Bottom
        auto bottom = [=]() -> void {
            auto row = 0u;
            for (size_t col = 1; col < *numCols - 1; col++) {
                const int cellIdx = row * (numCols * NumSpeeds) + col * NumSpeeds;
                const auto n = in[cellIdx + northCellOffset + SpeedIndexes::North];
                const auto ne = in[cellIdx + northEastCellOffset + SpeedIndexes::NorthEast];
                const auto nw = in[cellIdx + northWestCellOffset + SpeedIndexes::NorthWest];
                const auto w = in[cellIdx + westCellOffset + SpeedIndexes::West];
                const auto m = in[cellIdx + middleCellOffset + SpeedIndexes::Middle];
                const auto e = in[cellIdx + eastCellOffset + SpeedIndexes::East];
                const auto s = haloBottom[col * NumSpeeds + middleCellOffset + SpeedIndexes::South];
                const auto se = haloBottom[col * NumSpeeds + eastCellOffset + SpeedIndexes::SouthEast];
                const auto sw = haloBottom[col * NumSpeeds + westCellOffset + SpeedIndexes::SouthWest];

                float2 result[5] = {{m,  e},
                                    {n,  w},
                                    {s,  ne},
                                    {nw, sw},
                                    {se, 0.0f}};
                assignToCell(cellIdx, result);
            }
        };
        // Left
        auto left = [=]() -> void {
            auto col = 0u;
            for (size_t row = 1; row < *numRows - 1; row++) {
                const auto cellIdx = row * (numCols * NumSpeeds) + col * NumSpeeds;
                const auto n = in[cellIdx + northCellOffset + SpeedIndexes::North];
                const auto ne = in[cellIdx + northEastCellOffset + SpeedIndexes::NorthEast];
                const auto nw = haloLeft[row * NumSpeeds + sideHaloNorthCellOffset + SpeedIndexes::NorthWest];
                const auto w = haloLeft[row * NumSpeeds + middleCellOffset + SpeedIndexes::West];
                const auto m = in[cellIdx + middleCellOffset + SpeedIndexes::Middle];
                const auto e = in[cellIdx + eastCellOffset + SpeedIndexes::East];
                const auto s = in[cellIdx + southCellOffset + SpeedIndexes::South];
                const auto se = in[cellIdx + southEastCellOffset + SpeedIndexes::SouthEast];
                const auto sw = haloLeft[row * NumSpeeds + sideHaloSouthCellOffset + SpeedIndexes::SouthWest];

                float2 result[5] = {{m,  e},
                                    {n,  w},
                                    {s,  ne},
                                    {nw, sw},
                                    {se, 0.0f}};
                assignToCell(cellIdx, result);
            }
        };

        // Right
        auto right = [=]() -> void {
            auto col = numCols - 1;
            for (size_t row = 1; row < *numRows - 1; row++) {
                const auto cellIdx = row * (numCols * NumSpeeds) + col * NumSpeeds;
                const auto n = in[cellIdx + northCellOffset + SpeedIndexes::North];
                const auto ne = haloRight[row * NumSpeeds + sideHaloNorthCellOffset + SpeedIndexes::NorthEast];
                const auto nw = in[cellIdx + northWestCellOffset + SpeedIndexes::NorthWest];
                const auto w = in[cellIdx + westCellOffset + SpeedIndexes::West];
                const auto m = in[cellIdx + middleCellOffset + SpeedIndexes::Middle];
                const auto e = haloRight[row * NumSpeeds + middleCellOffset + SpeedIndexes::East];
                const auto s = in[cellIdx + southCellOffset + SpeedIndexes::South];
                const auto se = haloRight[row * NumSpeeds + sideHaloSouthCellOffset + SpeedIndexes::SouthEast];
                const auto sw = in[cellIdx + southWestCellOffset + SpeedIndexes::SouthWest];

                float2 result[5] = {{m,  e},
                                    {n,  w},
                                    {s,  ne},
                                    {nw, sw},
                                    {se, 0.0f}};
                assignToCell(cellIdx, result);
            }
        };

        // Middle
        auto middle = [=]() -> void {
            for (size_t row = 1; row < *numRows - 1; row++) {
                for (size_t col = 1; col < *numCols - 1; col++) {
                    const int cellIdx = row * (numCols * NumSpeeds) + col * NumSpeeds;

                    const auto n = in[cellIdx + northCellOffset + SpeedIndexes::North];
                    const auto ne = in[cellIdx + northEastCellOffset + SpeedIndexes::NorthEast];
                    const auto nw = in[cellIdx + northWestCellOffset + SpeedIndexes::NorthWest];
                    const auto w = in[cellIdx + westCellOffset + SpeedIndexes::West];
                    const auto m = in[cellIdx + middleCellOffset + SpeedIndexes::Middle];
                    const auto e = in[cellIdx + eastCellOffset + SpeedIndexes::East];
                    const auto s = in[cellIdx + southCellOffset + SpeedIndexes::South];
                    const auto se = in[cellIdx + southEastCellOffset + SpeedIndexes::SouthEast];
                    const auto sw = in[cellIdx + southWestCellOffset + SpeedIndexes::SouthWest];

                    float2 result[5] = {{m,  e},
                                        {n,  w},
                                        {s,  ne},
                                        {nw, sw},
                                        {se, 0.0f}};
                    assignToCell(cellIdx, result);

                }
            }
        };
        topLeft();
        top();
        topRight();
        left();
        right();
        middle();
        bottomLeft();
        bottom();
        bottomRight();


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
        const float c_sq = 1.f / 3.f; /* square of speed of sound */
        const float w0 = 4.f / 9.f;  /* weighting factor */
        const float w1 = 1.f / 9.f;  /* weighting factor */
        const float w2 = 1.f / 36.f; /* weighting factor */

        /* loop over the cells in the grid
        ** NB the collision step is called after
        ** the propagate step and so values of interest
        ** are in the scratch-space grid */
        for (int jj = 0; jj < *numRows; jj++) {
            for (int ii = 0; ii < *numCols; ii++) {
                auto idx = ii + jj * *numCols;
                /* don't consider occupied cells */
                // TODO can just fold rebound in here.
                if (!obstacles[idx]) {
                    /* compute local density total */
                    float local_density = 0.f;

                    for (int kk = 0; kk < NumSpeeds; kk++) {
                        local_density += in[idx + kk];
                    }

                    /* compute x velocity component */
                    float u_x = ((in[idx + 1] + in[idx + 5] + in[idx + 8])
                                 - (in[idx + 3] + in[idx + 6] + in[idx + 7]))
                                / local_density;
                    /* compute y velocity component */
                    float u_y = ((in[idx + 2] + in[idx + 5] + in[idx + 6])
                                 - (in[idx + 4] + in[idx + 7] + in[idx + 8])) / local_density;

                    /* velocity squared */
                    float u_sq = u_x * u_x + u_y * u_y;

                    /* directional velocity components */
                    float u[NumSpeeds];
                    u[1] = u_x;        /* east */
                    u[2] = u_y;  /* north */
                    u[3] = -u_x;        /* west */
                    u[4] = -u_y;  /* south */
                    u[5] = u_x + u_y;  /* north-east */
                    u[6] = -u_x + u_y;  /* north-west */
                    u[7] = -u_x - u_y;  /* south-west */
                    u[8] = u_x - u_y;  /* south-east */

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
                    //TODO can float2 this (or better yet float4)
                    for (int kk = 0; kk < NumSpeeds; kk++) {
                        out[idx + kk] = in[idx + kk] + omega * (d_equ[kk] - in[idx + kk]);
                    }
                }
            }
        }
        return true;
    }
};
