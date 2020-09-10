#include <poplar/Vertex.hpp>
#include <cstddef>
#include <print.h>
#include <math.h>
#include <ipudef.h>
#include <ipuvectormath>

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

struct Cell {
    float nw, n, ne, w, m, e, sw, s, se;
};


inline auto isObstacle(const Cell &cell) -> bool {
    return isnan(cell.m);
}


inline auto accelerateCell(Cell &cell, const float density, const float accel) -> void {
    const float w1 = density * accel / 9.f;
    const float w2 = density * accel / 36.f;

    const auto acceleratingWillMakeWestDensitiesNegative = [=](const Cell &cell) -> bool {
        return cell.e > w1 && cell.ne > w2 && cell.nw > w2;
    };

    if (!(isObstacle(cell) || acceleratingWillMakeWestDensitiesNegative(cell))) {
        cell.nw -= w2;
        cell.ne += w2;
        cell.w -= w1;
        cell.sw -= w2;
        cell.se += w2;
    }
}

class AccelerateFlowVertex : public Vertex { // can fold into progagate

public:
    InOut <Vector<float>> cellsInSecondRow; // 9 speeds in every cell, no halo
    Input<unsigned> partitionWidth;
    Input<float> density;
    Input<float> accel;

    bool compute() {
        float w1 = *density * *accel / 9.f;
        float w2 = *density * *accel / 36.f;

        for (auto col = 0u; col < *partitionWidth; col++) {
            auto cellOffset = NumSpeeds * col;
            accelerateCell(*reinterpret_cast<Cell *>(&cellsInSecondRow[cellOffset]), density, accel);
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


inline auto accelerateNeighbours(bool accelerateWest, bool accelerateMiddle, bool accelerateEast,
                                 Cell &nw, Cell &n, Cell &ne,
                                 Cell &w, Cell &m, Cell &e, Cell &sw, Cell &s,
                                 Cell &se) {
    if (accelerateWest) {
        accelerateCell(nw, density, accel);
        accelerateCell(w, density, accel);
        accelerateCell(sw, density, accel);
    }
    if (accelerateMiddle) {
        accelerateCell(n, density, accel);
        accelerateCell(m, density, accel);
        accelerateCell(s, density, accel);
    }
    if (accelerateEast) {
        accelerateCell(ne, density, accel);
        accelerateCell(e, density, accel);
        accelerateCell(se, density, accel);
    }
}


inline auto stream(const Cell &nw, const Cell &n, const Cell &ne,
                   const Cell &w, const Cell &m, const Cell &e, const Cell &sw, const Cell &s,
                   const Cell &se, Cell &out) -> void {
    out.nw = se.nw;
    out.n = s.n;
    out.ne = sw.ne;
    out.w = e.w;
    out.m = m.m;
    out.e = w.e;
    out.sw = sw.nw;
    out.s = se.nw;
    out.se = se.nw;
}


auto bgkCollision(const float omega, const Cell &in, Cell &out) -> float {
    /* compute local density total */
    auto local_density = in.nw + in.n + in.e + in.w + in.m + in.e + in.sw + in.s + in.se;

    float2 u = {
            ((in.ne + in.e + in.se) - (in.nw + in.w + in.sw)),
            ((in.nw + in.n + in.ne) - (in.sw + in.s + in.se))
    };
    u /= localDensity;

    /* velocity squared */
    const auto u_sq = (u * u)[0] + (u * u)[1];
    const auto u_over_2csq = u_sq / (2.f * c_sq);

    auto equ_calc = [](float weight, float2 directionalVelocity) -> float2 {
        return weight * local_density *
               (1.f + directionalVelocity / c_sq + (directionalVelocity * directionalVelocity) / cc2 - u_over_2csq);
    }

    float d_equ_m = w0 * local_density * (1.f - u_over_2csq);
    float2 d_equ_n_e = equ_calc(w1, {u[1], u[0]});
    float2 d_equ_s_w = equ_calc(w1, {-u[1], -u[0]});
    float2 d_equ_nw_ne = equ_calc(r2, {u[1] - u[0], u[1] + u[0]});
    float2 d_equ_sw_se = equ_calc(r2, {-u[1] - u[0], -u[1] + u[0]});

    out.m = in.m + omega * d_equ_m - cells.m;
    float2 n_e = float2{in.n, in.e} + o * d_equ_n_e - float2{in.n, in.e};
    float2 s_w = float2{in.s, in.w} + o * d_equ_s_w - float2{in.s, in.w};
    float2 nw_ne = float2{in.nw, in.ne} + o * d_equ_nw_ne - float2{in.nw, in.ne};
    float2 sw_se = float2{in.sw, in.se} + o * d_equ_sw_se - float2{in.sw, in.se};

    out.n = n_e[0];
    out.e = n_e[1];
    out.s = s_w[0];
    out.w = s_w[1];
    out.nw = nw_ne[0];
    out.ne = nw_ne[1];
    out.sw = sw_se[0];
    out.se = sw_se[1];

    return sqrtf(u_sq);
}


// optim obstacle totally enclosed -> (m= Nan and  s = Nan), do nothing

auto rebound(const Cell &in, Cell &out) -> bool {
    // We only do  this for obstacle locations, and we know it's an obstacle when middle is s NaN
    if isObstacle(in)
    {
        out.nw = in.se;
        out.n = in.s;
        out.ne = in.sw;
        out.w = in.e;
        out.m = in.m;
        out.e = in.w;
        out.sw = in.ne;
        out.s = in.n;
        out.se = in.nw;
    } else return Cell(in);
}

/* propagate densities from neighbouring cells, following
           ** appropriate directions of travel and writing into
           ** scratch space grid */
class D2Q9Vertex : public Vertex {

public:
    Input <Vector<float>> haloTop; // numCols x NumSpeeds
    Input <Vector<float>> haloLeft; // numRows x NumSpeeds
    Input <Vector<float>> haloRight; // numRows x NumSpeeds
    Input <Vector<float>> haloBottom; // numCols x NumSpeeds
    Input <Vector<float>> haloTopLeft; // 1x NumSpeeds
    Input <Vector<float>> haloTopRight;
    Input <Vector<float>> haloBottomLeft;
    Input <Vector<float>> haloBottomRight;

    Input <Vector<float>> in; // numCols x numRows x NumSpeeds
    unsigned numRows; // i.e. excluding halo
    unsigned numCols; // i.e. excluding halo
    float omega; // i.e. excluding halo
    float accel; // i.e. excluding halo
    int colToAccelerate; // column to accelerate, if any (-1 otherwise)

    Output <Vector<float>> out; // numCols x numRows x NumSpeeds

    bool compute() {
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
                const auto northCell = [=]() -> Cell {
                    if (row == TOP) {
                        return reinterpret_cast<Cell *>(haloTop[col * NumSpeeds + middleCellOffset])[0];
                    }
                    return reinterpret_cast<Cell *>(in[cellIdx + northCellOffset])[0];
                };

                const auto northEastCell = [=]() -> Cell {
                    if (row == TOP && col == RIGHT) {
                        return reinterpret_cast<Cell *>(&haloTopRight[0])[0];
                    } else if (row == TOP) {
                        return reinterpret_cast<Cell *>(haloTop[col * NumSpeeds + eastCellOffset])[0];
                    } else if (col == RIGHT) {
                        return reinterpret_cast<Cell *>(haloRight[row * NumSpeeds + sideHaloNorthCellOffset])[0];
                    }
                    return reinterpret_cast<Cell *>(in[cellIdx + northEastCellOffset])[0];
                };

                const auto northWestCellSouthEastSpeed = [=]() -> Cell {
                    if (row == TOP && col == LEFT) {
                        return *haloTopLeft;
                    } else if (row == TOP) {
                        return haloTop[col * NumSpeeds + westCellOffset + SpeedIndexes::SouthEast];
                    } else if (col == LEFT) {
                        return haloLeft[row * NumSpeeds + sideHaloNorthCellOffset + SpeedIndexes::SouthEast];
                    }
                    return in[cellIdx + northWestCellOffset + SpeedIndexes::SouthEast];
                };

                const auto westCellEastSpeed = [=]() -> Cell {
                    if (col == LEFT) {
                        return haloLeft[row * NumSpeeds + middleCellOffset + SpeedIndexes::East];
                    }
                    return in[cellIdx + westCellOffset + SpeedIndexes::East];
                };

                const auto middle = [=]() -> Cell {
                    return in[cellIdx + middleCellOffset + SpeedIndexes::Middle];
                };

                const auto eastCellWestSpeed = [=]() -> Cell {
                    if (col == RIGHT) {
                        return haloRight[row * NumSpeeds + middleCellOffset + SpeedIndexes::West];
                    }
                    return in[cellIdx + eastCellOffset + SpeedIndexes::West];
                };

                const auto southCellNorthSpeed = [=]() -> Cell {
                    if (row == BOTTOM) {
                        return haloBottom[col * NumSpeeds + middleCellOffset + SpeedIndexes::North];
                    }
                    return in[cellIdx + southCellOffset + SpeedIndexes::North];
                };

                const auto southEastCellNorthWestSpeed = [=]() -> Cell {
                    if (row == BOTTOM && col == RIGHT) {
                        return *haloBottomRight;
                    } else if (row == BOTTOM) {
                        return haloBottom[col * NumSpeeds + eastCellOffset + SpeedIndexes::NorthWest];
                    } else if (col == RIGHT) {
                        return haloRight[row * NumSpeeds + sideHaloSouthCellOffset + SpeedIndexes::NorthWest];
                    }
                    return in[cellIdx + southEastCellOffset + SpeedIndexes::NorthWest];
                };


                const auto southWestCellNorthEastSpeed = [=]() -> Cell {
                    if (row == BOTTOM && col == LEFT) {
                        return *haloBottomLeft;
                    } else if (row == BOTTOM) {
                        return haloBottom[col * NumSpeeds + westCellOffset + SpeedIndexes::NorthEast];
                    } else if (col == LEFT) {
                        return haloLeft[row * NumSpeeds + sideHaloSouthCellOffset + SpeedIndexes::NorthEast];
                    }
                    return in[cellIdx + southWestCellOffset + SpeedIndexes::NorthEast];
                };

                accelerateMiddle = col == colToAccelerate;
                accelerateWest = colToAccelerate > 0 && col == colToAccelerate + 1;
                accelerateRight = colToAccelerate > 0 && col == colToAccelerate - 1;

                Cell nw = northWestCell();
                Cell n = northCell();
                Cell ne = northEastCell();
                Cell w = westCell();
                Cell m = middleCell();
                Cell e = eastCell();
                Cell sw = southWestCell();
                Cell s = southCell();
                Cell se = southEastCell();

                accelerateNeighbours(accelerateWest, accelerateMiddle, accelerateEast,
                                     northWestCell, northCell, northEastCell,
                                     westCell, middleCell, eastCell,
                                     southWestCell, southCell, southEastCell);
                Cell updatedCell;
                stream(northWestCell, northCell, northEastCell,
                       westCell, middleCell, eastCell,
                       southWestCell, southCell, southEastCell, updatedCell);
                if (!rebound(updatedCell)) {
                    bgkCollision(omega, in, out, velocity);
                }
            }
        }
        return true;
    }
};




