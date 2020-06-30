#include <poplar/Vertex.hpp>
#include <cstddef>
#include <print.h>
#include <math.h>
#include <ipudef.h>

using namespace poplar;

constexpr auto NumSpeeds = 9u;

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

        auto cellAddress = reinterpret_cast<float2 *>(&cells[0]);

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

class PropagateVertex : public Vertex {

public:
    Input <Vector<float>> in; // 9 speeds in every cell - this includes the halo
    Input<unsigned> numRows; // Including halo
    Input<unsigned> numCols; // Including halo
    Input <Vector<float>> out; // 9 speeds in every cell - this includes the halo

    bool compute() {
        for (size_t jj = 1; jj < *numRows - 1; jj++) // don't loop through halo
        {
            for (size_t ii = 1; ii < *numCols - 1; ii++) {
                auto cellIdx = jj * (numCols * NumSpeeds) + ii * NumSpeeds;
                // We don't have to worry about wraparound and edges because the input vector already has the halo
                auto northIdx = cellIdx + (numCols * NumSpeeds); // Remember layout is (0,0) = bottom left
                auto southIdx = cellIdx - (numCols * NumSpeeds);
                auto eastIdx = cellIdx + NumSpeeds;
                auto westIdx = cellIdx - NumSpeeds;
                auto nwIdx = northIdx - NumSpeeds;
                auto neIdx = northIdx + NumSpeeds;
                auto seIdx = southIdx + NumSpeeds;
                auto swIdx = southIdx - NumSpeeds;

                /* propagate densities from neighbouring cells, following
                ** appropriate directions of travel and writing into
                ** scratch space grid */
                out[cellIdx] = in[cellIdx]; /* central cell, no movement */
                out[cellIdx + 1] = in[eastIdx + 1]; /* east */
                out[cellIdx + 2] = in[northIdx + 2]; /* north */
                out[cellIdx + 3] = in[westIdx + 3]; /* west */
                out[cellIdx + 4] = in[southIdx + 4]; /* south */
                out[cellIdx + 5] = in[neIdx + 5]; /* north-east */
                out[cellIdx + 6] = in[nwIdx + 6]; /* north-west */
                out[cellIdx + 7] = in[swIdx + 7]; /* south-west */
                out[cellIdx + 8] = in[seIdx + 8]; /* south-east */
            }
        }
    }
};

//class ReboundVertex : public Vertex {
//
//public:
//    Input <Vector<float>> in; // 9 speeds in every cell, no halo
//    Input <Vector<float>> out; // 9 speeds in every cell, no halo
//    Input <Vector<bool>> obstacles; //  no halo
//    Input<unsigned> numRows; // no halo
//    Input<unsigned> numCols; // no halo
//    Output<float> out;
//
//    bool compute() {
//        /* loop over the cells in the grid */
//        for (size_t jj = 0; jj < *numRows; jj++) {
//#pragma unroll 4
//            for (size_t ii = 0; ii < *numCols; ii++) {
//                auto obstacleIdx = jj * numCols + ii;
//                auto cellIdx = jj * (numCols * NumSpeeds) + (ii * NumSpeeds);
//                /* if the cell contains an obstacle */
//                if (obstacles[obstacleIdx]) {
//                    /* called after propagate, so taking values from scratch space
//                    ** mirroring, and writing into main grid */
//                    out[cellIdx + 1] = in[cellIdx + 3];
//                    out[cellIdx + 2] = in[cellIdx + 4];
//                    out[cellIdx + 3] = in[cellIdx + 1];
//                    out[cellIdx + 4] = in[cellIdx + 2];
//                    out[cellIdx + 5] = in[cellIdx + 7];
//                    out[cellIdx + 6] = in[cellIdx + 8];
//                    out[cellIdx + 7] = in[cellIdx + 5];
//                    out[cellIdx + 8] = in[cellIdx + 6];
//                }
//            }
//        }
//    }
//};