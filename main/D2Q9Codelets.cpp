#include <poplar/Vertex.hpp>
#include <cstddef>

using namespace poplar;

constexpr auto NumSpeeds = 9u;

class MaskedSumPartial : public Vertex { // On each worker, reduce the cells

public:
    Input <Vector<float>> cells; // 9 speeds in every cell
    Input <Vector<bool>> obstacles;
    Input<unsigned> numCells;
    Output<float> out;

    bool compute() {
        const auto cellContribution = [this](int i) -> float {
            auto mask = obstacles[i];
            auto s = 0.0f;
#pragma unroll 9
            for (int j = 0; j < 9; j++) {
                s += cells[i * 9 + j];
            }
            return mask * s;
        };
        auto tmp = 0.0f;
        for (auto i = 0; i < *numCells; i++) {
            tmp += cellContribution(i);
        }
        *out = tmp;

        return true;
    }
};


class ReducePartials : public Vertex { // Take the partials within a tile and reduce them
public:
    Input <Vector<float>> partials;
    Input<unsigned> numPartials;
    Output<float> out;

    bool compute() {
        auto tmp = 0.0f;
        for (int i = 0; i < *numPartials; i++)
            tmp += partials[i];
        *out = tmp;
        return true;
    }
};

class AppendReducedSum : public Vertex { // Reduce the per-tile partial sums and append to the average list

public:
    Input <Vector<float>> partials;
    Input<unsigned> index;
    Input<unsigned> numPartials;
    Output <Vector<float>> finals;

    bool compute() {

        auto tmp = 0.0f;
        for (auto i = 0; i < *numPartials; i++) {
            tmp += partials[i];
        }
        finals[*index] = tmp;

        return true;
    }
};
//
//class AccelerateFlowVertex : public Vertex {
//
//public:
//    Input <Vector<float>> cellsInSecondRow; // 9 speeds in every cell, no halo
//    Input <Vector<bool>> obstaclesInSecondRow;
//    Input<unsigned> partitionWidth;
//    Input<float> density;
//    Input<float> accel;
//
//    bool compute() {
//        float w1 = *density * *accel / 9.f;
//        float w2 = *density * *accel / 36.f;
//        for (auto col = 0u; col < *partitionWidth; col++) {
//            auto cellOffset = NumSpeeds * col;
//
//            /* if the cell is not occupied and we don't send a negative density */
//            if (!obstaclesInSecondRow[col]
//                && (cellsInSecondRow[cellOffset + 3] - w1) > 0.f
//                && (cellsInSecondRow[cellOffset + 6] - w2) > 0.f
//                && (cellsInSecondRow[cellOffset + 7] - w2) > 0.f) {
//                /* increase 'east-side' densities */
//                cellsInSecondRow[cellOffset + 1] += w1;
//                cellsInSecondRow[cellOffset + 5] += w2;
//                cellsInSecondRow[cellOffset + 8] += w2;
//                /* decrease 'west-side' densities */
//                cellsInSecondRow[cellOffset + 3] -= w1;
//                cellsInSecondRow[cellOffset + 6] -= w2;
//                cellsInSecondRow[cellOffset + 7] -= w2;
//            }
//        }
//        return true;
//    }
//};
//
//class PropagateVertex : public Vertex {
//
//public:
//    Input <Vector<float>> in; // 9 speeds in every cell - this includes the halo
//    Input<unsigned> numRows; // Including halo
//    Input<unsigned> numCols; // Including halo
//    Input <Vector<float>> out; // 9 speeds in every cell - this includes the halo
//
//    bool compute() {
//        for (size_t jj = 1; jj < *numRows - 1; jj++) // don't loop through halo
//        {
//            for (size_t ii = 1; ii < *numCols - 1; ii++) {
//                auto cellIdx = jj * (numCols * NumSpeeds) + ii * NumSpeeds;
//                // We don't have to worry about wraparound and edges because the input vector already has the halo
//                auto northIdx = cellIdx + (numCols * NumSpeeds); // Remember layout is (0,0) = bottom left
//                auto southIdx = cellIdx - (numCols * NumSpeeds);
//                auto eastIdx = cellIdx + NumSpeeds;
//                auto westIdx = cellIdx - NumSpeeds;
//                auto nwIdx = northIdx - NumSpeeds;
//                auto neIdx = northIdx + NumSpeeds;
//                auto seIdx = southIdx + NumSpeeds;
//                auto swIdx = southIdx - NumSpeeds;
//
//                /* propagate densities from neighbouring cells, following
//                ** appropriate directions of travel and writing into
//                ** scratch space grid */
//                out[cellIdx] = in[cellIdx]; /* central cell, no movement */
//                out[cellIdx + 1] = in[eastIdx + 1]; /* east */
//                out[cellIdx + 2] = in[northIdx + 2]; /* north */
//                out[cellIdx + 3] = in[westIdx + 3]; /* west */
//                out[cellIdx + 4] = in[southIdx + 4]; /* south */
//                out[cellIdx + 5] = in[neIdx + 5]; /* north-east */
//                out[cellIdx + 6] = in[nwIdx + 6]; /* north-west */
//                out[cellIdx + 7] = in[swIdx + 7]; /* south-west */
//                out[cellIdx + 8] = in[seIdx + 8]; /* south-east */
//            }
//        }
//    }
//};
//
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