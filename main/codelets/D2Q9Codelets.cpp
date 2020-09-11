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
enum Speed {
    Middle, East, North, West, South, NorthEast, NorthWest, SouthWest, SouthEast
};

/**
 *  The output array of summed velocities is spread throughout the distributed memories. We give each
 *  tile a vertex that knows what bits of the array are mapped to it. The sumOfVelocities is broadcast to
 *  each tile, and only the tile owning the memory writes it.
 */
class AppendReducedSum : public Vertex { // Reduce the per-tile partial sums and append to the list of sums

public:
    Input<float> sumOfVelocities;
    Input<unsigned> indexToWrite;
    unsigned myStartIndex; // The index where my array starts
    unsigned myEndIndex; // My last index (inclusive)
    InOut <Vector<float>> finals; // The piece of the array I have

    bool compute() {

        const auto idx = *indexToWrite;
        if ((idx >= myStartIndex) && (idx <= myEndIndex)) {
            finals[idx - myStartIndex] = sumOfVelocities;
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
    unsigned width;
    float w1;
    float w2;

    bool compute() {
        for (auto col = 0u; col < width; col++) {
            /* if the cell is not occupied and we don't send a negative density */
            if (!obstaclesInSecondRow[col]
                && (cellsInSecondRow[col * NumSpeeds + Speed::West] > w1)
                && (cellsInSecondRow[col * NumSpeeds + Speed::NorthWest] > w2)
                && (cellsInSecondRow[col * NumSpeeds + Speed::SouthWest] > w2)) {
                /* increase 'east-side' densities */
                cellsInSecondRow[col * NumSpeeds + Speed::East] += w1;
                cellsInSecondRow[col * NumSpeeds + Speed::NorthEast] += w2;
                cellsInSecondRow[col * NumSpeeds + Speed::SouthEast] += w2;
                /* decrease 'west-side' densities */
                cellsInSecondRow[col * NumSpeeds + Speed::West] -= w1;
                cellsInSecondRow[col * NumSpeeds + Speed::SouthWest] -= w2;
                cellsInSecondRow[col * NumSpeeds + Speed::NorthWest] -= w2;

            }
        }
        return true;
    }
};
//
//class CollisionVertex : public Vertex {
//
//public:
//    Input <Vector<float, VectorLayout::ONE_PTR>> in; // 9 speeds in every cell, no halo
//    Input <Vector<bool, VectorLayout::ONE_PTR>> obstacles; //  no halo
//    Output <Vector<float, VectorLayout::ONE_PTR>> out;
//    Output<float> normedVelocityPartial; // sum of normed velocities for non-obstacle cells
//
//    unsigned height;
//    unsigned width;
//    float omega;
//
//    bool compute() {
//        const float *inPtr = reinterpret_cast<float *>(&in[0]); // Since we're using ONE_PTR, contiguous, we can
//        // encourage better vectorisation by using plain old array refs
//        float *outPtr = reinterpret_cast<float *>(&out[0]);
//        auto tmp_velocityPartial = 0.f;
//
//        constexpr auto c_sq = 1.f / 3.f; /* square of speed of sound */
//        constexpr auto cc2 = (2.f * c_sq * c_sq);
//        constexpr auto w0 = 4.f / 9.f;  /* weighting factor */
//        constexpr auto w1 = 1.f / 9.f;  /* weighting factor */
//        constexpr auto w2 = 1.f / 36.f; /* weighting factor */
//
//        /* loop over the cells in the grid
//        ** NB the collision step is called after
//        ** the propagate step and so values of interest
//        ** are in the scratch-space grid */
//
//        for (int y = 0; y < height; y++) {
//            for (int x = 0; x < width; x++) {
//                const auto rebound = obstacles OBS_OFFSET(0, 0);
//                if (rebound) {
//                    outPtr ATSPEED(Speed::North) = inPtr ATSPEED(Speed::South);
//                    outPtr ATSPEED(Speed::South) = inPtr ATSPEED(Speed::North);
//                    outPtr ATSPEED(Speed::West) = inPtr ATSPEED(Speed::East);
//                    outPtr ATSPEED(Speed::East) = inPtr ATSPEED(Speed::West);
//                    outPtr ATSPEED(Speed::NorthWest) = inPtr ATSPEED(Speed::SouthEast);
//                    outPtr ATSPEED(Speed::SouthEast) = inPtr ATSPEED(Speed::NorthWest);
//                    outPtr ATSPEED(Speed::NorthEast) = inPtr ATSPEED(Speed::SouthWest);
//                    outPtr ATSPEED(Speed::SouthWest) = inPtr ATSPEED(Speed::NorthEast);
//                } else {
//
//
//                    float newVal[NumSpeeds];
//                    newVal[Speed::Middle] = inPtr ATSPEED(Speed::Middle);
//                    newVal[Speed::North] = inPtr ATSPEED(Speed::North);
//                    newVal[Speed::South] = inPtr ATSPEED(Speed::South);
//                    newVal[Speed::East] = inPtr ATSPEED(Speed::East);
//                    newVal[Speed::West] = inPtr ATSPEED(Speed::West);
//                    newVal[Speed::NorthWest] = inPtr ATSPEED(Speed::NorthWest);
//                    newVal[Speed::NorthEast] = inPtr ATSPEED(Speed::NorthEast);
//                    newVal[Speed::SouthWest] = inPtr ATSPEED(Speed::SouthWest);
//                    newVal[Speed::SouthEast] = inPtr ATSPEED(Speed::SouthEast);
//
//                    /* compute local density total */
//                    auto local_density = 0.f;
//
//                    for (int kk = 0; kk < NumSpeeds; kk++) {
//                        local_density += inPtr ATSPEED(kk);
//                    }
//
//                    /* compute x velocity component */
//                    const auto u_x = ((inPtr ATSPEED(Speed::East) +
//                                       inPtr ATSPEED(Speed::NorthEast) +
//                                       inPtr ATSPEED(Speed::SouthEast)
//                                      ) -
//                                      (inPtr ATSPEED(Speed::West) +
//                                       inPtr ATSPEED(Speed::SouthWest) +
//                                       inPtr ATSPEED(Speed::NorthWest)
//                                      )) / local_density;
//                    /* compute y velocity component */
//                    const auto u_y = ((inPtr ATSPEED(Speed::NorthWest) +
//                                       inPtr ATSPEED(Speed::North) +
//                                       inPtr ATSPEED(Speed::NorthEast)) -
//                                      (inPtr ATSPEED (Speed::SouthWest) +
//                                       inPtr ATSPEED(Speed::South) +
//                                       inPtr ATSPEED(Speed::SouthEast))
//                                     ) / local_density;
//
//                    /* velocity squared */
//                    const auto u_sq = u_x * u_x + u_y * u_y;
//                    const auto u_over_2csq = u_sq / (2.f * c_sq);
//
//                    auto equilibriumDensity = [&](const float weight, const float speed) -> float {
//                        return weight * local_density * (1.f + speed / c_sq + (speed * speed) / cc2 - u_over_2csq);
//                    };
//
//                    auto relaxed = [&](const float i, const float weight, const float speed) -> float {
//                        return i + omega * (equilibriumDensity(weight, speed) - i);
//                    };
//
//                    //     float newVal[NumSpeeds];
//                    newVal[Speed::Middle] = relaxed(inPtr ATSPEED(Speed::Middle), w0, 0);
//                    newVal[Speed::North] = relaxed(inPtr ATSPEED(Speed::North), w1, u_y);
//                    newVal[Speed::South] = relaxed(inPtr ATSPEED(Speed::South), w1, -u_y);
//                    newVal[Speed::East] = relaxed(inPtr ATSPEED(Speed::East), w1, u_x);
//                    newVal[Speed::West] = relaxed(inPtr ATSPEED(Speed::West), w1, -u_x);
//                    newVal[Speed::NorthWest] = relaxed(inPtr ATSPEED(Speed::NorthWest), w2, u_y - u_x);
//                    newVal[Speed::NorthEast] = relaxed(inPtr ATSPEED(Speed::NorthEast), w2, u_y + u_x);
//                    newVal[Speed::SouthWest] = relaxed(inPtr ATSPEED(Speed::SouthWest), w2, -u_y - u_x);
//                    newVal[Speed::SouthEast] = relaxed(inPtr ATSPEED(Speed::SouthEast), w2, -u_y + u_x);
//                    auto new_local_density = 0;
//                    for (int i = 0; i < NumSpeeds; i++) {
//                        new_local_density += newVal[i];
//                    }
//                    if (new_local_density == 0)new_local_density = 1;
//                    /* compute x velocity component */
//                    const auto nu_x = ((newVal[Speed::East] +
//                                        newVal[Speed::NorthEast] +
//                                        newVal[Speed::SouthEast]
//                                       ) -
//                                       (newVal[Speed::West] +
//                                        newVal[Speed::SouthWest] +
//                                        newVal[Speed::NorthWest]
//                                       )) / new_local_density;
//                    /* compute y velocity component */
//                    const auto nu_y = ((newVal[Speed::NorthWest] +
//                                        newVal[Speed::North] +
//                                        newVal[Speed::NorthEast]) -
//                                       (newVal[Speed::SouthWest] +
//                                        newVal[Speed::South] +
//                                        newVal[Speed::SouthEast])
//                                      ) / new_local_density;
//
//                    /* velocity squared */
//                    const auto nu_sq = nu_x * nu_x + nu_y * nu_y;
//                    tmp_velocityPartial += sqrtf(nu_sq);
//                    for (int i = 0; i < NumSpeeds; i++) {
//                        outPtr ATSPEED(i) = newVal[i];
//                    }
//                }
//            }
//        }
//        *normedVelocityPartial = tmp_velocityPartial;
//        return true;
//    }
//};


/* propagate densities from neighbouring cells, following
           ** appropriate directions of travel and writing into
           ** scratch space grid */
#define OFFSET(r, c) (y + r + 1) * (width + 2) + (c + x + 1)

class PropagateVertexFloatSoA : public Vertex {
public:
    // Input speeds (with halos): (width +2) x (height +2)
    Input <Vector<float>> mSpeeds, nSpeeds, sSpeeds, eSpeeds, wSpeeds, nwSpeeds, neSpeeds, swSpeeds, seSpeeds; //
    // Output speeds (no halos):width x height
    Output <Vector<float>> mSpeedsOut, nSpeedsOut, sSpeedsOut, eSpeedsOut, wSpeedsOut, nwSpeedsOut, neSpeedsOut, swSpeedsOut, seSpeedsOut; //
    unsigned width;
    unsigned height;

    bool compute() {
        for (auto y = 0u; y < height; y++) {
            for (auto x = 0u; x < width; x++) {
                // Copy North neighbour's South speed etc
                const auto idx = y * width + x;
                mSpeedsOut[idx] = mSpeeds[OFFSET(0, 0)];
                sSpeedsOut[idx] = sSpeeds[OFFSET(1, 0)];
                nSpeedsOut[idx] = nSpeeds[OFFSET(-1, 0)];
                eSpeedsOut[idx] = eSpeeds[OFFSET(0, -1)];
                wSpeedsOut[idx] = wSpeeds[OFFSET(0, 1)];
                neSpeedsOut[idx] = neSpeeds[OFFSET(-1, -1)];
                seSpeedsOut[idx] = seSpeeds[OFFSET(1, -1)];
                swSpeedsOut[idx] = swSpeeds[OFFSET(1, 1)];
                nwSpeedsOut[idx] = nwSpeeds[OFFSET(-1, 1)];
            }
        }
        return true;
    }
};


/** Recast Input/Output as a float2 * to generate 64-bit loads and stores */
#define AS_F4(X)    reinterpret_cast<float4 *>(&X[0])
/** The index in the float2 array of the current (x,y) index offset by (R,C) items */
#define F4_OUTIDX  (quarterWidth * y + x)
#define F4_INIDX(ROW, COL)     ((quarterWidth +2)* ((y + 1) + ROW) + ((x + 1) + COL))

class PropagateVertexFloat4SoA : public Vertex {
public:
    // Input speeds (with halos): (width +2) x (height +2)
    Input <Vector<float, VectorLayout::ONE_PTR, 8>> mSpeeds, nSpeeds, sSpeeds, eSpeeds, wSpeeds, nwSpeeds, neSpeeds, swSpeeds, seSpeeds; //
    // Output speeds (no halos):width x height
    Output <Vector<float, VectorLayout::ONE_PTR, 8>> mSpeedsOut, nSpeedsOut, sSpeedsOut, eSpeedsOut, wSpeedsOut, nwSpeedsOut, neSpeedsOut, swSpeedsOut, seSpeedsOut; //
    unsigned width;
    unsigned height;

    bool compute() {
        // Caveat: width must be divisible by 4!
        const unsigned quarterWidth = width / 4.0f;
        for (auto y = 0u; y < height; y++) {
            for (auto x = 0u; x < quarterWidth; x++) {
                // Copy North neighbour's South speed etc
                AS_F4(mSpeedsOut)[F4_OUTIDX] = AS_F4(mSpeeds)[F4_INIDX(0, 0)];
                AS_F4(sSpeedsOut)[F4_OUTIDX] = AS_F4(sSpeeds)[F4_INIDX(1, 0)];
                AS_F4(nSpeedsOut)[F4_OUTIDX] = AS_F4(nSpeeds)[F4_INIDX(-1, 0)];
                AS_F4(eSpeedsOut)[F4_OUTIDX] = AS_F4(eSpeeds)[F4_INIDX(0, -1)];
                AS_F4(wSpeedsOut)[F4_OUTIDX] = AS_F4(wSpeeds)[F4_INIDX(0, 1)];
                AS_F4(neSpeedsOut)[F4_OUTIDX] = AS_F4(neSpeeds)[F4_INIDX(-1, -1)];
                AS_F4(seSpeedsOut)[F4_OUTIDX] = AS_F4(seSpeeds)[F4_INIDX(1, -1)];
                AS_F4(swSpeedsOut)[F4_OUTIDX] = AS_F4(swSpeeds)[F4_INIDX(1, 1)];
                AS_F4(nwSpeedsOut)[F4_OUTIDX] = AS_F4(nwSpeeds)[F4_INIDX(-1, 1)];
            }
        }
        return true;
    }
};


/** Recast Input/Output as a float2 * to generate 64-bit loads and stores */
#define AS_F2(X)    reinterpret_cast<float2 *>(&X[0])
/** The index in the float2 array of the current (x,y) index offset by (R,C) items */
#define F2_OUTIDX  (halfWidth * y + x)
#define F2_INIDX(ROW, COL)     ((halfWidth +2)* ((y + 1) + ROW) + ((x + 1) + COL))

class PropagateVertexFloat2SoA : public Vertex {
public:
    // Input speeds (with halos): (width +2) x (height +2)
    Input <Vector<float, VectorLayout::ONE_PTR, 8>> mSpeeds, nSpeeds, sSpeeds, eSpeeds, wSpeeds, nwSpeeds, neSpeeds, swSpeeds, seSpeeds; //
    // Output speeds (no halos):width x height
    Output <Vector<float, VectorLayout::ONE_PTR, 8>> mSpeedsOut, nSpeedsOut, sSpeedsOut, eSpeedsOut, wSpeedsOut, nwSpeedsOut, neSpeedsOut, swSpeedsOut, seSpeedsOut; //
    unsigned width;
    unsigned height;

    bool compute() {
        // Caveat: width must be divisible by 4!
        const unsigned halfWidth = width / 2.0f;
        for (auto y = 0u; y < height; y++) {
            for (auto x = 0u; x < halfWidth; x++) {
                // Copy North neighbour's South speed etc
                AS_F2(mSpeedsOut)[F2_OUTIDX] = AS_F2(mSpeeds)[F2_INIDX(0, 0)];
                AS_F2(sSpeedsOut)[F2_OUTIDX] = AS_F2(sSpeeds)[F2_INIDX(1, 0)];
                AS_F2(nSpeedsOut)[F2_OUTIDX] = AS_F2(nSpeeds)[F2_INIDX(-1, 0)];
                AS_F2(eSpeedsOut)[F2_OUTIDX] = AS_F2(eSpeeds)[F2_INIDX(0, -1)];
                AS_F2(wSpeedsOut)[F2_OUTIDX] = AS_F2(wSpeeds)[F2_INIDX(0, 1)];
                AS_F2(neSpeedsOut)[F2_OUTIDX] = AS_F2(neSpeeds)[F2_INIDX(-1, -1)];
                AS_F2(seSpeedsOut)[F2_OUTIDX] = AS_F2(seSpeeds)[F2_INIDX(1, -1)];
                AS_F2(swSpeedsOut)[F2_OUTIDX] = AS_F2(swSpeeds)[F2_INIDX(1, 1)];
                AS_F2(nwSpeedsOut)[F2_OUTIDX] = AS_F2(nwSpeeds)[F2_INIDX(-1, 1)];
            }
        }
        return true;
    }
};


struct Cell {
    float speeds[9];
};

inline auto lbmKernel(const Cell *__restrict nw, const Cell *__restrict n, const Cell *__restrict ne,
                      const Cell *__restrict w, const Cell *__restrict m, const Cell *__restrict e,
                      const Cell *__restrict sw, const Cell *__restrict s, const Cell *__restrict se,
                      const bool isObstacle, const bool isAccelerate,
                      const float omega, const float oneMinusOmega,
                      const float w1, const float w2) -> Cell {

    Cell result;
    // Streaming
    const float speed_nw = se->speeds[Speed::NorthWest];
    const float speed_n = s->speeds[Speed::North];
    const float speed_ne = sw->speeds[Speed::NorthEast];
    const float speed_w = e->speeds[Speed::West];
    const float speed_m = m->speeds[Speed::Middle];
    const float speed_e = w->speeds[Speed::East];
    const float speed_sw = ne->speeds[Speed::SouthWest];
    const float speed_s = n->speeds[Speed::South];
    const float speed_se = nw->speeds[Speed::SouthEast];

    if (isObstacle) {
        // rebound
        result.speeds[Speed::NorthWest] = speed_se;
        result.speeds[Speed::North] = speed_s;
        result.speeds[Speed::NorthEast] = speed_sw;
        result.speeds[Speed::West] = speed_e;
        result.speeds[Speed::Middle] = speed_m;
        result.speeds[Speed::East] = speed_w;
        result.speeds[Speed::SouthWest] = speed_ne;
        result.speeds[Speed::South] = speed_n;
        result.speeds[Speed::SouthEast] = speed_nw;

    } else {
        // collision (with acceleration folded in)
        const float local_density = speed_nw + speed_n + speed_ne +
                                    speed_w + speed_m + speed_e +
                                    speed_sw + speed_s + speed_se;
        /* compute x velocity component */
        const float u_x = (speed_e + speed_ne + speed_se - (speed_w + speed_ne + speed_sw)) / local_density;
        /* compute y velocity component */
        const float u_y = (speed_n + speed_nw + speed_ne - (speed_s + speed_sw + speed_se)) / local_density;
        const float u_sq = u_x * u_x + u_y * u_y;
        const float c_sq = 1.00f - u_sq * 1.50f;
        const float ld0 = 4.00f / 9.00f * local_density * omega;
        const float ld1 = local_density / 9.00f * omega;
        const float ld2 = local_density / 36.00f * omega;

        const auto relax = [&](const float &speed, const float &weight, const float &velocityComponent) -> float {
            return speed * oneMinusOmega +
                   weight * ((4.50f * velocityComponent) * (2.00f / 3.00f + velocityComponent) + c_sq);
        };

        const auto relax2 = [&](const float2 speed, const float weight, const float2 velocityComponent) -> float2 {
            return speed * oneMinusOmega +
                   weight * ((4.50f * velocityComponent) * (2.00f / 3.00f + velocityComponent) + c_sq);
        };


        auto[speed_out_w, speed_out_e] = relax2(float2{speed_w, speed_e}, ld1, float2{-u_x, u_x});
        auto[speed_out_s, speed_out_n] = relax2(float2{speed_s, speed_n}, ld1, float2{-u_y, u_y});
        auto[speed_out_nw, speed_out_ne] = relax2(float2{speed_nw, speed_ne}, ld2, float2{-u_x + u_y, u_x + u_y});
        auto[speed_out_sw, speed_out_se] = relax2(float2{speed_sw, speed_se}, ld2, float2{-u_x - u_y, u_x - u_y});

        const float speed_out_m = relax(speed_m, ld0, 0);
//        const float speed_out_e = relax(speed_e, ld1, +u_x);
//        const float speed_out_n = relax(speed_n, ld1, +u_y);
//        const float speed_out_w = relax(speed_w, ld1, -u_x);
//        const float speed_out_s = relax(speed_s, ld1, -u_y);
//        const float speed_out_ne = relax(speed_ne, ld2, u_x + u_y);
//        const float speed_out_nw = relax(speed_nw, ld2, -u_x + u_y);
//        const float speed_out_sw = relax(speed_sw, ld2, -u_x - u_y);
//        const float speed_out_se = relax(speed_se, ld2, u_x - u_y);

        if (isAccelerate) {
            result.speeds[Speed::NorthWest] = speed_out_nw - w2;
            result.speeds[Speed::North] = speed_out_n;
            result.speeds[Speed::NorthEast] = speed_out_ne + w2;
            result.speeds[Speed::West] = speed_out_w - w1;
            result.speeds[Speed::Middle] = speed_out_m;
            result.speeds[Speed::East] = speed_out_e + w1;
            result.speeds[Speed::SouthWest] = speed_out_sw - w2;
            result.speeds[Speed::South] = speed_out_s;
            result.speeds[Speed::SouthEast] = speed_out_se + w2;
        } else {
            result.speeds[Speed::NorthWest] = speed_out_nw;
            result.speeds[Speed::North] = speed_out_n;
            result.speeds[Speed::NorthEast] = speed_out_ne;
            result.speeds[Speed::West] = speed_out_w;
            result.speeds[Speed::Middle] = speed_out_m;
            result.speeds[Speed::East] = speed_out_e;
            result.speeds[Speed::SouthWest] = speed_out_sw;
            result.speeds[Speed::South] = speed_out_s;
            result.speeds[Speed::SouthEast] = speed_out_se;
        }
    }
    return result;
}


inline auto normedLocalSpeed(const Cell &c) -> float {
    const float local_density = c.speeds[Speed::NorthWest] + c.speeds[Speed::North] + c.speeds[Speed::NorthEast] +
                                c.speeds[Speed::West] + c.speeds[Speed::Middle] + c.speeds[Speed::East] +
                                c.speeds[Speed::SouthWest] + c.speeds[Speed::South] + c.speeds[Speed::SouthEast];
    /* compute x velocity component */
    const float u_x = (c.speeds[Speed::East] + c.speeds[Speed::NorthEast] + c.speeds[Speed::SouthEast] -
                       (c.speeds[Speed::West] + c.speeds[Speed::NorthWest] + c.speeds[Speed::NorthEast])) /
                      local_density;
    /* compute y velocity component */
    const float u_y = (c.speeds[Speed::North] + c.speeds[Speed::NorthWest] + c.speeds[Speed::NorthEast] -
                       (c.speeds[Speed::South] + c.speeds[Speed::SouthWest] + c.speeds[Speed::SouthEast])) /
                      local_density;
    return sqrtf(u_x * u_x + u_y * u_y);
}

#define HALO_OFFSET(ROW, COL, speed)   9 * ((width +2)* ((y + 1) + ROW) + ((x + 1) + COL)) + speed
#define CELL_AT(r, c) reinterpret_cast<const Cell *>(&in[HALO_OFFSET(r, c, 0)]);

class LbmVertex : public Vertex {

public:
    Input <Vector<float, VectorLayout::ONE_PTR>> in; // (width +2) x (height +2) x NumSpeeds
    Input <Vector<bool, VectorLayout::ONE_PTR>> obstacles; // width x height
    Output <Vector<float, VectorLayout::ONE_PTR>> out; // width x height x NumSpeeds
    Output<float> normedVelocityPartial;
    unsigned width;
    unsigned height;
    unsigned rowToAccelerate;
    unsigned isAcceleratingVertex;
    float omega;
    float oneMinusOmega;
    float w1; // density * accel / 9.0f;
    float w2; // density * accel / 36.0f;

    bool compute() {
        auto v = 0.f;
        for (auto y = 0u; y < height; y++) {
            const bool isAccelerate = isAcceleratingVertex && y == rowToAccelerate;
            for (auto x = 0u; x < width; x++) {
                const auto idx = 9 * (x + y * width);
                Cell *outPtr = reinterpret_cast<Cell *>(&out[idx]);
                const bool isObstacle = obstacles[idx];

                const Cell *nw = CELL_AT(1, -1);
                const Cell *n = CELL_AT(1, 0);
                const Cell *ne = CELL_AT(1, 1);
                const Cell *w = CELL_AT(0, 1);
                const Cell *m = CELL_AT(0, 0);
                const Cell *e = CELL_AT(0, 1);
                const Cell *sw = CELL_AT(1, -1);
                const Cell *s = CELL_AT(1, 0);
                const Cell *se = CELL_AT(1, 1);

                const auto result = lbmKernel(nw, n, ne, w, m, e, sw, s, se, isObstacle,
                                              isAccelerate, omega, oneMinusOmega, w1, w2);

//                Cell result2 = {
//                        .speeds = {1,(float)x,(float)y,1,1,1,1,1,1}
//                };
                v += normedLocalSpeed(result);
                *outPtr = result;

            }
        }
        *normedVelocityPartial = v;
        return true;
    }
};


/**
 * propagate densities from neighbouring cells, following
* appropriate directions of travel and writing into
* scratch space grid */

class PropagateVertexAoS : public Vertex {

public:
    Input <Vector<float, VectorLayout::ONE_PTR>> in; // (width +2) x (height +2) x NumSpeeds
    Output <Vector<float, VectorLayout::ONE_PTR>> out; // width x height x NumSpeeds
    unsigned width; // of this worker's partition
    unsigned height; // of this worker's partition

    bool compute() {

        for (auto y = 0u; y < height; y++) {
            for (auto x = 0u; x < width; x++) {
                auto o = reinterpret_cast<float *>(&out[x + y * width]);
                o[Speed::Middle] = in[HALO_OFFSET (0, 0, Speed::Middle)];
                o[Speed::East] = in[HALO_OFFSET(0, -1, Speed::East)];
                o[Speed::West] = in[HALO_OFFSET(0, +1, Speed::West)];
                o[Speed::South] = in[HALO_OFFSET(-1, 0, Speed::South)];
                o[Speed::North] = in[HALO_OFFSET(1, 0, Speed::North)];
                o[Speed::NorthWest] = in[HALO_OFFSET(-1, -1, Speed::NorthWest)];
                o[Speed::NorthEast] = in[HALO_OFFSET(-1, 1, Speed::NorthEast)];
                o[Speed::SouthWest] = in[HALO_OFFSET(1, -1, Speed::SouthWest)];
                o[Speed::SouthEast] = in[HALO_OFFSET(1, 1, Speed::SouthEast)];
            }
        }
        return true;
    }
};

struct Params {
    int ny;
    int nx;
    int maxIters;
    float omega;
    float one_minus_omega;
    float density;
    float accel;
    int rowToAccelerate;
    bool isAccelerate;
    int total_free_cells;
};

void firstAccel(const Params &params, const bool *obstacles, Cell *cells);
auto nuevo(const Params &params, const Cell *cells_old, Cell *cells_new, const bool *obstacles) -> float;



void firstAccel(const Params &params, const bool *obstacles, Cell *cells) {
    const float w1 = params.density * params.accel / 9.f;
    const float w2 = params.density * params.accel / 36.f;
    for (int ii = 0; ii < params.nx; ii++)
    {
        /* if the cell is not occupied and
        ** we don't send a negative density */
        if (!obstacles[ii]
            && (cells[ii].speeds[3] - w1) > 0.f
            && (cells[ii].speeds[6] - w2) > 0.f
            && (cells[ii].speeds[7] - w2) > 0.f)
        {
            /* increase 'east-side' densities */
            cells[ii].speeds[1] += w1;
            cells[ii].speeds[5] += w2;
            cells[ii].speeds[8] += w2;
            /* decrease 'west-side' densities */
            cells[ii].speeds[3] -= w1;
            cells[ii].speeds[6] -= w2;
            cells[ii].speeds[7] -= w2;
        }
    }
}
auto nuevo(const Params &params, const Cell *cells_old, Cell *cells_new, const bool *obstacles)  -> float{
    /* compute weighting factors */
    const float w1 = params.density * params.accel / 9.f;
    const float w2 = params.density * params.accel / 36.f;
    float tot_u = 0.00f;

    // Old is ny rows x nx cols
#define NEW_OFFSET(r, c) (ii  + c) + (jj  + r) * params.nx
#define OLD_OFFSET(r, c) (ii +1 + c) + (jj  + 1 + r) * (params.nx + 2)
    for (int jj = 0; jj < params.ny; jj++)
    {
        for (int ii = 0; ii < params.nx; ii++)
        {
            const int y_n = (jj + 1) % params.ny;
            const int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);
            const float accel = jj == params.rowToAccelerate && params.isAccelerate ? 1.f : 0.f;
            const int x_e = (ii + 1) % params.nx;
            const int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
            const int is_obstacle = obstacles[NEW_OFFSET(0, 0)];


            const float speeds_0 = cells_old[OLD_OFFSET(+0, +0)].speeds[0]; /* central cell, no movement */
            const float speeds_1 = cells_old[OLD_OFFSET(+0, -1)].speeds[1]; /* east */
            const float speeds_2 = cells_old[OLD_OFFSET(-1, +0)].speeds[2]; /* north */
            const float speeds_3 = cells_old[OLD_OFFSET(+0, +1)].speeds[3]; /* west */
            const float speeds_4 = cells_old[OLD_OFFSET(+1, +0)].speeds[4]; /* south */
            const float speeds_5 = cells_old[OLD_OFFSET(-1, -1)].speeds[5]; /* north-east */
            const float speeds_6 = cells_old[OLD_OFFSET(-1, +1)].speeds[6]; /* north-west */
            const float speeds_7 = cells_old[OLD_OFFSET(+1, +1)].speeds[7]; /* south-west */
            const float speeds_8 = cells_old[OLD_OFFSET(+1, -1)].speeds[8]; /* south-east */

            // const float speeds_0 = cells_old[ii + jj * params.nx].speeds[0]; /* central cell, no movement */
            // const float speeds_1 = cells_old[x_w + jj * params.nx].speeds[1]; /* east */
            // const float speeds_2 = cells_old[ii + y_s * params.nx].speeds[2]; /* north */
            // const float speeds_3 = cells_old[x_e + jj * params.nx].speeds[3]; /* west */
            // const float speeds_4 = cells_old[ii + y_n * params.nx].speeds[4]; /* south */
            // const float speeds_5 = cells_old[x_w + y_s * params.nx].speeds[5]; /* north-east */
            // const float speeds_6 = cells_old[x_e + y_s * params.nx].speeds[6]; /* north-west */
            // const float speeds_7 = cells_old[x_e + y_n * params.nx].speeds[7]; /* south-west */
            // const float speeds_8 = cells_old[x_w + y_n * params.nx].speeds[8]; /* south-east */

            if (is_obstacle)
            {
                cells_new[NEW_OFFSET(0,0)].speeds[0] = speeds_0;
                cells_new[NEW_OFFSET(0,0)].speeds[1] = speeds_3;
                cells_new[NEW_OFFSET(0,0)].speeds[2] = speeds_4;
                cells_new[NEW_OFFSET(0,0)].speeds[3] = speeds_1;
                cells_new[NEW_OFFSET(0,0)].speeds[4] = speeds_2;
                cells_new[NEW_OFFSET(0,0)].speeds[5] = speeds_7;
                cells_new[NEW_OFFSET(0,0)].speeds[6] = speeds_8;
                cells_new[NEW_OFFSET(0,0)].speeds[7] = speeds_5;
                cells_new[NEW_OFFSET(0,0)].speeds[8] = speeds_6;

                
            }
            else {

                /* compute local density total */
                const float local_density =
                    speeds_0 + speeds_1 + speeds_2 + speeds_3 + speeds_4 + speeds_5 + speeds_6 + speeds_7 +
                    speeds_8;

                /* compute x velocity component */
                const float u_x = (speeds_1 + speeds_5 + speeds_8 - (speeds_3 + speeds_6 + speeds_7)) / local_density;
                /* compute y velocity component */
                const float u_y = (speeds_2 + speeds_5 + speeds_6 - (speeds_4 + speeds_7 + speeds_8)) / local_density;

                /* velocity squared */
                const float u_sq = u_x * u_x + u_y * u_y;

                const float c_sq = 1.00f - u_sq * 1.50f;
                const float ld0 = 4.00f / 9.00f * local_density * params.omega;
                const float ld1 = local_density / 9.00f * params.omega;
                const float ld2 = local_density / 36.00f * params.omega;
                const float u_s = u_x + u_y;
                const float u_d = -u_x + u_y;

                const float speeds_out_0 = speeds_0 * params.one_minus_omega + ld0 * c_sq;
                const float speeds_out_1 =
                    speeds_1 * params.one_minus_omega + ld1 * ((4.50f * u_x) * (2.00f / 3.00f + u_x) + c_sq);
                const float speeds_out_2 =
                    speeds_2 * params.one_minus_omega + ld1 * ((4.50f * u_y) * (2.00f / 3.00f + u_y) + c_sq);
                const float speeds_out_3 =
                    speeds_3 * params.one_minus_omega + ld1 * ((-4.50f * u_x) * (2.00f / 3.00f - u_x) + c_sq);
                const float speeds_out_4 =
                    speeds_4 * params.one_minus_omega + ld1 * ((-4.50f * u_y) * (2.00f / 3.00f - u_y) + c_sq);
                const float speeds_out_5 =
                    speeds_5 * params.one_minus_omega + ld2 * ((4.50f * u_s) * (2.00f / 3.00f + u_s) + c_sq);
                const float speeds_out_6 =
                    speeds_6 * params.one_minus_omega + ld2 * ((4.50f * u_d) * (2.00f / 3.00f + u_d) + c_sq);
                const float speeds_out_7 =
                    speeds_7 * params.one_minus_omega + ld2 * ((-4.50f * u_s) * (2.00f / 3.00f - u_s) + c_sq);
                const float speeds_out_8 =
                    speeds_8 * params.one_minus_omega + ld2 * ((-4.50f * u_d) * (2.00f / 3.00f - u_d) + c_sq);

                cells_new[NEW_OFFSET(0,0)].speeds[0] = speeds_out_0;
                cells_new[NEW_OFFSET(0,0)].speeds[1] = speeds_out_1 + accel * w1;
                cells_new[NEW_OFFSET(0,0)].speeds[2] = speeds_out_2;
                cells_new[NEW_OFFSET(0,0)].speeds[3] = speeds_out_3 - accel * w1;
                cells_new[NEW_OFFSET(0,0)].speeds[4] = speeds_out_4;
                cells_new[NEW_OFFSET(0,0)].speeds[5] = speeds_out_5 + accel * w2;
                cells_new[NEW_OFFSET(0,0)].speeds[6] = speeds_out_6 - accel * w2;
                cells_new[NEW_OFFSET(0,0)].speeds[7] = speeds_out_7 - accel * w2;
                cells_new[NEW_OFFSET(0,0)].speeds[8] = speeds_out_8 + accel * w2;
                tot_u += sqrtf(u_sq);
            }
        }
    }
    return tot_u / (float)params.total_free_cells;
}

class FirstAccelVertex : public Vertex
{

public:
    InOut<Vector<float, VectorLayout::ONE_PTR>> cellsVec;
    Input<Vector<bool, VectorLayout::ONE_PTR>> obstaclesVec;
    int nx;
    float density;
    float accel;

    bool compute()
    {
        auto cells = reinterpret_cast<Cell *>(&cellsVec[0]);
        auto obstacles = reinterpret_cast<bool *>(&obstaclesVec[0]);

        auto params = Params{
            .ny = 0,
            .nx = nx,
            .maxIters = 0,
            .omega = 0,
            .one_minus_omega = 0,
            .density = density,
            .accel = accel,
            .isAccelerate = true,
            .rowToAccelerate = 0,
            .total_free_cells = 0};

        firstAccel(params, obstacles, cells);

        return true;
    }
};

class LastHopeVertex : public Vertex
{

public:
    Input<Vector<float, VectorLayout::ONE_PTR>> cells_oldVec;
    Output<Vector<float, VectorLayout::ONE_PTR>> cells_newVec;
    Input<Vector<bool, VectorLayout::ONE_PTR>> obstaclesVec;
    Output<float> av_vel;
    int ny;
    int nx;
    int maxIters;
    float omega;
    float one_minus_omega;
    float density;
    float accel;
    float iter;
    int total_free_cells;
    bool isAccelerate;
    int rowToAccelerate;

    bool compute() {
        auto cells_old = reinterpret_cast<Cell *>(&cells_oldVec[0]);
        auto cells_new = reinterpret_cast<Cell *>(&cells_newVec[0]);
        auto obstacles = reinterpret_cast<bool *>(&obstaclesVec[0]);

        auto params = Params {
            .ny = ny,
            .nx = nx,
            .maxIters = maxIters,
            .omega = omega,
            .one_minus_omega = one_minus_omega,
            .density = density,
            .accel = accel,
            .isAccelerate = isAccelerate,
            .rowToAccelerate = rowToAccelerate,
            .total_free_cells = total_free_cells
        };

        *av_vel = nuevo(params, cells_old, cells_new, obstacles);
    
        return true;
    }
};

/*
Total compute time was         4.274s
Reynolds number:        1.541786193848e+02
HOST total density: 1.024019470215e+02
*/

