#include <poplar/Vertex.hpp>
#include <cstddef>
#include <print.h>
#include <math.h>
#include <ipudef.h>

using namespace poplar;


//------------------------------------- FLOAT2 Implementation ---------------------------------------------------------
/**
 * Perform the Gaussian blur on 2 channels (or pixels of the same channel) simultaneously
 */
float2 stencil(const float2 nw, const float2 n, const float2 ne, const float2 w, const float2 m,
               const float2 e, const float2 sw,
               const float2 s, const float2 se) {
    return 1.f / 16 * (nw + ne + sw + se) + 4.f / 16 * m + 2.f / 16 * (e + w + s + n);
}

/** Recast Input/Output as a float2 * to generate 64-bit loads and stores */
#define AS_F2(X)    reinterpret_cast<float2 *>(&X[0])
/** The index in the float2 array of the current (x,y) index offset by (R,C) items */
#define F2_IDX(R, C) 2 * (nx * (y + R) + (x + C)) + c
/** Cell offset in a halo (e.g. one to the right in the north halo) */
#define F2_HALO_OFFSET_H(dx) (x + dx) * 2 + c
/** Cell offset in a halo (e.g. one to the bottom in the west halo) */
#define F2_HALO_OFFSET_V(dy) (y + dy) * 2 + c

/**
 * The float2 vectorised general case, where the worker's partition is >= 2x2, and we blur the RG and then the BA channels
 * in 2 vectorised steps. In this case there are 9 different loop regions,
 * which have to take into account that cell neighbours may be from different data structures (the halos). The
 * regions are: (as region name: [data structures used]):
 * 1. Top Left: nw, n, w, m
 * 2. Top: n, m
 * 3. Top Right: ne, n, e, m
 * 4. Left: w, m
 * 5. Middle: m
 * 6. Right: e, m
 * 7. Bottom Left: sw, s, w, m
 * 8. Bottom: s, m
 * 9. Bottom Right: se, s, e, m
 */
class GaussianBlurCodeletFloat2 : public Vertex {

public:
    Input <Vector<float, VectorLayout::ONE_PTR, 8>> in;
    Input <Vector<float, VectorLayout::ONE_PTR, 8>> nw, ne, sw, se;
    Input <Vector<float, VectorLayout::ONE_PTR, 8>> n, s, w, e;
    Output <Vector<float, VectorLayout::ONE_PTR, 8>> out;
    unsigned width;
    unsigned height;

    bool compute() {
        const auto nx = width;
        const auto ny = height;

        const auto f2in = AS_F2(in);
        auto f2out = AS_F2(out);

//         Only works if this is at least a 2x2 block (excluding halos), and in must be same size as out
        if (nx > 1 && ny > 1) {
            // top left
            {
                constexpr auto x = 0u;
                constexpr auto y = 0u;
#pragma unroll 2
                for (auto c = 0u; c < 2; c++) {
                    const auto _nw = AS_F2(nw)[c];
                    const auto _w = AS_F2(w)[F2_HALO_OFFSET_V(0)];
                    const auto _sw = AS_F2(w)[F2_HALO_OFFSET_V(1)];
                    const auto _n = AS_F2(n)[F2_HALO_OFFSET_H(0)];
                    const auto _m = f2in[F2_IDX(0, 0)];
                    const auto _s = f2in[F2_IDX(1, 0)];
                    const auto _ne = AS_F2(n)[F2_HALO_OFFSET_H(1)];
                    const auto _e = f2in[F2_IDX(0, 1)];
                    const auto _se = f2in[F2_IDX(1, 1)];
                    f2out[F2_IDX(0, 0)] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
                }
            }

            // top
            {
                constexpr auto y = 0u;
                for (auto x = 1u; x < nx - 1; x++) {
#pragma unroll 2
                    for (auto c = 0; c < 2; c++) {
                        const auto _nw = AS_F2(n)[F2_HALO_OFFSET_H(-1)];
                        const auto _w = f2in[F2_IDX(0, -1)];
                        const auto _sw = f2in[F2_IDX(1, -1)];
                        const auto _n = AS_F2(n)[F2_HALO_OFFSET_H(0)];
                        const auto _m = f2in[F2_IDX(0, 0)];
                        const auto _s = f2in[F2_IDX(1, 0)];
                        const auto _ne = AS_F2(n)[F2_HALO_OFFSET_H(1)];
                        const auto _e = f2in[F2_IDX(0, 1)];
                        const auto _se = f2in[F2_IDX(1, 1)];
                        f2out[F2_IDX(0, 0)] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
                    }
                }
            }

            // top right
            {
                const auto x = nx - 1u;
                constexpr auto y = 0u;
#pragma unroll 2
                for (auto c = 0; c < 2; c++) {
                    const auto _nw = AS_F2(n)[F2_HALO_OFFSET_H(-1)];
                    const auto _w = f2in[F2_IDX(0, -1)];
                    const auto _sw = f2in[F2_IDX(1, -1)];
                    const auto _n = AS_F2(n)[F2_HALO_OFFSET_H(0)];
                    const auto _m = f2in[F2_IDX(0, 0)];
                    const auto _s = f2in[F2_IDX(1, 0)];
                    const auto _ne = AS_F2(ne)[c];
                    const auto _e = AS_F2(e)[F2_HALO_OFFSET_V(0)];
                    const auto _se = AS_F2(e)[F2_HALO_OFFSET_V(1)];
                    f2out[F2_IDX(0, 0)] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
                }
            }

            // left col
            {
                constexpr auto x = 0u;
                for (auto y = 1; y < ny - 1; y++) {
#pragma unroll 2
                    for (auto c = 0; c < 2; c++) {
                        const auto _nw = AS_F2(w)[F2_HALO_OFFSET_V(-1)];
                        const auto _w = AS_F2(w)[F2_HALO_OFFSET_V(0)];
                        const auto _sw = AS_F2(w)[F2_HALO_OFFSET_V(1)];
                        const auto _n = f2in[F2_IDX(-1, 0)];
                        const auto _m = f2in[F2_IDX(0, 0)];
                        const auto _s = f2in[F2_IDX(1, 0)];
                        const auto _ne = f2in[F2_IDX(-1, 1)];
                        const auto _e = f2in[F2_IDX(0, 1)];
                        const auto _se = f2in[F2_IDX(1, 1)];
                        f2out[F2_IDX(0, 0)] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
                    }
                }
            }

            // middle block
            for (auto y = 1; y < ny - 1; y++) {
                for (auto x = 1; x < nx - 1; x++) {
#pragma unroll 2
                    for (auto c = 0u; c < 2; c++) {
                        const auto _nw = f2in[F2_IDX(-1, -1)];
                        const auto _w = f2in[F2_IDX(0, -1)];
                        const auto _sw = f2in[F2_IDX(+1, -1)];
                        const auto _n = f2in[F2_IDX(-1, 0)];
                        const auto _m = f2in[F2_IDX(0, 0)];
                        const auto _s = f2in[F2_IDX(+1, 0)];
                        const auto _ne = f2in[F2_IDX(+1, +1)];
                        const auto _e = f2in[F2_IDX(0, +1)];
                        const auto _se = f2in[F2_IDX(-1, +1)];
                        f2out[F2_IDX(0, 0)] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
                    }
                }
            }

            // right col
            {
                const auto x = nx - 1u;
                for (auto y = 1; y < ny - 1u; y++) {
#pragma unroll 2
                    for (auto c = 0; c < 2; c++) {
                        const auto _nw = f2in[F2_IDX(-1, -1)];
                        const auto _w = f2in[F2_IDX(0, -1)];
                        const auto _sw = f2in[F2_IDX(1, -1)];
                        const auto _n = f2in[F2_IDX(-1, 0)];
                        const auto _m = f2in[F2_IDX(0, 0)];
                        const auto _s = f2in[F2_IDX(1, 0)];
                        const auto _ne = AS_F2(e)[F2_HALO_OFFSET_V(-1)];
                        const auto _e = AS_F2(e)[F2_HALO_OFFSET_V(0)];
                        const auto _se = AS_F2(e)[F2_HALO_OFFSET_V(1)];
                        f2out[F2_IDX(0, 0)] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
                    }
                }
            }

            // bottom left
            {
                const auto y = ny - 1;
                constexpr auto x = 0u;
#pragma unroll 2
                for (auto c = 0; c < 2; c++) {
                    const auto _nw = AS_F2(w)[F2_HALO_OFFSET_V(-1)];
                    const auto _w = AS_F2(w)[F2_HALO_OFFSET_V(0)];
                    const auto _sw = AS_F2(sw)[c];
                    const auto _n = f2in[F2_IDX(-1, 0)];
                    const auto _m = f2in[F2_IDX(0, 0)];
                    const auto _s = AS_F2(s)[F2_HALO_OFFSET_H(0)];
                    const auto _ne = f2in[F2_IDX(-1, 1)];
                    const auto _e = f2in[F2_IDX(0, 1)];
                    const auto _se = AS_F2(s)[F2_HALO_OFFSET_H(1)];
                    f2out[F2_IDX(0, 0)] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
                }
            }

            // bottom
            {
                const auto y = ny - 1;
                for (auto x = 1u; x < nx - 1u; x++) {
#pragma unroll 2
                    for (auto c = 0; c < 2; c++) {
                        const auto _nw = f2in[F2_IDX(-1, -1)];
                        const auto _w = f2in[F2_IDX(0, -1)];
                        const auto _sw = AS_F2(s)[F2_HALO_OFFSET_H(-1)];
                        const auto _n = f2in[F2_IDX(-1, 0)];
                        const auto _m = f2in[F2_IDX(0, 0)];
                        const auto _s = AS_F2(s)[F2_HALO_OFFSET_H(0)];
                        const auto _ne = f2in[F2_IDX(-1, 1)];
                        const auto _e = f2in[F2_IDX(0, 1)];
                        const auto _se = AS_F2(s)[F2_HALO_OFFSET_H(1)];
                        f2out[F2_IDX(0, 0)] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
                    }
                }
            }

            // bottom right
            {
                const auto y = ny - 1;
                const auto x = nx - 1;
#pragma unroll 2
                for (auto c = 0; c < 2; c++) {
                    const auto _nw = f2in[F2_IDX(-1, -1)];
                    const auto _w = f2in[F2_IDX(0, -1)];
                    const auto _sw = AS_F2(s)[F2_HALO_OFFSET_H(-1)];
                    const auto _n = f2in[F2_IDX(-1, 0)];
                    const auto _m = f2in[F2_IDX(0, 0)];
                    const auto _s = AS_F2(s)[F2_HALO_OFFSET_H(0)];
                    const auto _ne = AS_F2(e)[F2_HALO_OFFSET_V(-1)];
                    const auto _e = AS_F2(e)[F2_HALO_OFFSET_V(0)];
                    const auto _se = AS_F2(se)[c];
                    f2out[F2_IDX(0, 0)] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
                }
            }
            return true;
        }
        return false;
    }
};


/**
 * The float2 vectorised version of the special case where the worker's partition 1 row high, but >1 cells wide.
 * We use the vectorised stencil to do the RG channels at the same time, then the BA channels
 * Now we have 3 special loop cases which have to take into account that cell neighbours may be from different
 * data structures (the halos). The
 * regions are: (as region name: [data structures used]):
 * 1. Left: nw, n, w, sw, s, m
 * 2. Middle: n, m, s
 * 3. Right: ne, n, e, se, s, m
 */
class GaussianWide1RowBlurCodeletFloat2 : public Vertex {

public:
    Input <Vector<float, VectorLayout::ONE_PTR, 8>> in;
    Input <Vector<float, VectorLayout::ONE_PTR, 8>> nw, ne, sw, se;
    Input <Vector<float, VectorLayout::ONE_PTR, 8>> n, s, w, e;
    Output <Vector<float, VectorLayout::ONE_PTR, 8>> out;
    unsigned width;
    unsigned height; // Unused, but we want to keep the interface the same

    bool compute() {
        const auto f2in = AS_F2(in);
        auto f2out = AS_F2(out);
        const auto nx = width;
        constexpr auto ny = 1;

//         Only works if this is at least a 1x2 block (excluding halos), and in must be same size as out
        constexpr auto y = 0u;
        if (height == 1 && nx > 1) {
            // Left
            {
                const auto x = 0u;
#pragma unroll 2
                for (auto c = 0u; c < 2; c++) {
                    const auto _nw = AS_F2(nw)[c];
                    const auto _w = AS_F2(w)[c];
                    const auto _sw = AS_F2(sw)[c];
                    const auto _n = AS_F2(n)[F2_HALO_OFFSET_H(0)];
                    const auto _m = f2in[F2_IDX(0, 0)];
                    const auto _s = AS_F2(s)[F2_HALO_OFFSET_H(0)];
                    const auto _ne = AS_F2(n)[F2_HALO_OFFSET_H(1)];
                    const auto _e = f2in[F2_IDX(0, 1)];
                    const auto _se = AS_F2(s)[F2_HALO_OFFSET_H(1)];
                    f2out[F2_IDX(0, 0)] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
                }
            }

            // Middle
            {
                for (auto x = 1u; x < nx - 1; x++) {
#pragma unroll 2
                    for (auto c = 0; c < 2; c++) {
                        const auto _nw = AS_F2(n)[F2_HALO_OFFSET_H(-1)];
                        const auto _w = f2in[F2_IDX(0, -1)];
                        const auto _sw = AS_F2(s)[F2_HALO_OFFSET_H(-1)];
                        const auto _n = AS_F2(n)[F2_HALO_OFFSET_H(0)];
                        const auto _m = f2in[F2_IDX(0, 0)];
                        const auto _s = AS_F2(s)[F2_HALO_OFFSET_H(0)];
                        const auto _ne = AS_F2(n)[F2_HALO_OFFSET_H(1)];
                        const auto _e = f2in[F2_IDX(0, 1)];
                        const auto _se = AS_F2(s)[F2_HALO_OFFSET_H(1)];
                        f2out[F2_IDX(0, 0)] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
                    }
                }
            }

            // Right
            {
                const auto x = nx - 1u;
#pragma unroll 2
                for (auto c = 0; c < 2; c++) {
                    const auto _nw = AS_F2(n)[F2_HALO_OFFSET_H(-1)];
                    const auto _w = f2in[F2_IDX(0, -1)];
                    const auto _sw = AS_F2(s)[F2_HALO_OFFSET_H(-1)];
                    const auto _n = AS_F2(n)[F2_HALO_OFFSET_H(0)];
                    const auto _m = f2in[F2_IDX(0, 0)];
                    const auto _s = AS_F2(s)[F2_HALO_OFFSET_H(0)];
                    const auto _ne = AS_F2(n)[F2_HALO_OFFSET_H(1)];
                    const auto _e = f2in[F2_IDX(0, 1)];
                    const auto _se = AS_F2(s)[F2_HALO_OFFSET_H(1)];
                    f2out[F2_IDX(0, 0)] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
                }
            }
            return true;
        }
        return false;
    }
};


/**
 * The float2 vectorised version of the special case where the worker's partition 1 column wide, but >1 cells high. Now we have 3 special loop cases
 * which have to take into account that cell neighbours may be from different data structures (the halos). We do the RG, then BA channels at the same time.
 * The regions are: (as region name: [data structures used]):
 * 1. Top: nw, n, ne, w, e, m
 * 2. Middle: w, m, e
 * 3. Bottom: sw, w, m, se, s, e
 */
template<typename T>
class GaussianNarrow1ColBlurCodeletFloat2 : public Vertex {

public:
    Input <Vector<float, VectorLayout::ONE_PTR, 8>> in;
    Input <Vector<float, VectorLayout::ONE_PTR, 8>> nw, ne, sw, se;
    Input <Vector<float, VectorLayout::ONE_PTR, 8>> n, s, w, e;
    Output <Vector<float, VectorLayout::ONE_PTR, 8>> out;
    unsigned width;
    unsigned height; // Unused, but we want to keep the interface the same

    bool compute() {
        const auto ny = height;
        constexpr auto nx = 1;
        const auto f2in = AS_F2(in);
        auto f2out = AS_F2(out);

//         Only works if this is at least a 2x1 block (excluding halos), and in must be same size as out
        if (width == 1 && ny > 1) {
            constexpr auto x = 0u;
            // Top
            {
                constexpr auto y = 0u;
#pragma unroll 2
                for (auto c = 0u; c < 2; c++) {
                    const auto _nw = AS_F2(nw)[c];
                    const auto _w = AS_F2(w)[F2_HALO_OFFSET_V(0)];
                    const auto _sw = AS_F2(w)[F2_HALO_OFFSET_V(1)];
                    const auto _n = AS_F2(n)[c];
                    const auto _m = f2in[F2_IDX(0, 0)];
                    const auto _s = f2in[F2_IDX(1, 0)];
                    const auto _ne = AS_F2(ne)[c];
                    const auto _e = AS_F2(e)[F2_HALO_OFFSET_V(0)];
                    const auto _se = AS_F2(e)[F2_HALO_OFFSET_V(1)];
                    f2out[F2_IDX(0, 0)] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
                }
            }

            // Middle
            {
                for (auto y = 1u; y < ny - 1; y++) {
#pragma unroll 2
                    for (auto c = 0; c < 2; c++) {
                        const auto _nw = AS_F2(w)[F2_HALO_OFFSET_V(-1)];
                        const auto _w = AS_F2(w)[F2_HALO_OFFSET_V(0)];
                        const auto _sw = AS_F2(w)[F2_HALO_OFFSET_V(1)];
                        const auto _n = f2in[F2_IDX(-1, 0)];
                        const auto _m = f2in[F2_IDX(0, 0)];
                        const auto _s = f2in[F2_IDX(1, 0)];
                        const auto _ne = AS_F2(e)[F2_HALO_OFFSET_V(-1)];
                        const auto _e = AS_F2(e)[F2_HALO_OFFSET_V(0)];
                        const auto _se = AS_F2(e)[F2_HALO_OFFSET_V(1)];
                        f2out[F2_IDX(0, 0)] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
                    }
                }
            }

            // Right
            {
                const auto y = ny - 1u;
#pragma unroll 2
                for (auto c = 0; c < 2; c++) {
                    const auto _nw = AS_F2(w)[F2_HALO_OFFSET_V(-1)];
                    const auto _w = AS_F2(w)[F2_HALO_OFFSET_V(0)];
                    const auto _sw = AS_F2(sw)[c];
                    const auto _n = f2in[F2_IDX(-1, 0)];
                    const auto _m = f2in[F2_IDX(0, 0)];
                    const auto _s = AS_F2(s)[c];
                    const auto _ne = AS_F2(e)[F2_HALO_OFFSET_V(-1)];
                    const auto _e = AS_F2(e)[F2_HALO_OFFSET_V(0)];
                    const auto _se = AS_F2(se)[c];
                    f2out[F2_IDX(0, 0)] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
                }
            }
            return true;
        }
        return false;
    }
};


/**
 * Vectorised float2 version of the extreme case, where the worker's partition 1 column wide and 1 cell high. Now all neighbours come from the
 * 8 halo regions. We do the RG channels together then the BA channels
 */
template<typename T>
class GaussianBlur1x1CodeletFloat2 : public Vertex { // Extreme case of a 1x1 partition!

public:
    Input <Vector<float, VectorLayout::ONE_PTR, 8>> in;
    Input <Vector<float, VectorLayout::ONE_PTR, 8>> nw, ne, sw, se;
    Input <Vector<float, VectorLayout::ONE_PTR, 8>> n, s, w, e;
    Output <Vector<float, VectorLayout::ONE_PTR, 8>> out;
    unsigned width;
    unsigned height; // Unused, but we want to keep the interface the same for easy re-use

    bool compute() {
        const auto ny = height;
        const auto nx = width;
        const auto f2in = AS_F2(in);
        auto f2out = AS_F2(out);

        constexpr auto x = 0u;
        constexpr auto y = 0u;
//         Only works if this is at least a 1x1 block (excluding halos), and in must be same size as out
        if (nx == 1 && ny > 1) {
#pragma unroll 2
            for (auto c = 0u; c < 2; c++) {
                const auto _nw = AS_F2(nw)[c];
                const auto _w = AS_F2(w)[c];
                const auto _sw = AS_F2(sw)[c];
                const auto _n = AS_F2(n)[c];
                const auto _m = f2in[c];
                const auto _s = AS_F2(s)[c];
                const auto _ne = AS_F2(ne)[c];
                const auto _e = AS_F2(e)[c];
                const auto _se = AS_F2(se)[c];
                f2out[F2_IDX(0, 0)] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
            }

            return true;
        }
        return false;
    }
};

//---------------------------- half4 implementation--------------------------------------------
half4 stencil(const half4 nw, const half4 n, const half4 ne, const half4 w, const half4 m,
              const half4 e, const half4 sw,
              const half4 s, const half4 se) {
    constexpr auto sixteenth = (half) 1.f / 16;
    constexpr auto eighth = (half) 2.f / 16;
    constexpr auto quarter = (half) 4.f / 16;
    return sixteenth * (nw + ne + sw + se) + quarter * m + eighth * (e + w + s + n);
}



/** Recast Input/Output as a half4 * to generate 64-bit loads and stores */
#define AS_H4(X)    reinterpret_cast<half4 *>(&X[0])
/** The index in the half4 array of the current (x,y) index offset by (R,C) items */
#define H4_IDX(R, C) (nx * (y + R) + (x + C))
/** Cell offset in a halo (e.g. one to the right in the north halo) */
#define H4_HALO_OFFSET_H(dx) (x + dx)
/** Cell offset in a halo (e.g. one to the bottom in the west halo) */
#define H4_HALO_OFFSET_V(dy) (y + dy)

/**
 * The half4 vectorised general case, where the worker's partition is >= 2x2, and we blur the RGBA channels
 * in 1 vectorised steps. In this case there are 9 different loop regions,
 * which have to take into account that cell neighbours may be from different data structures (the halos). The
 * regions are: (as region name: [data structures used]):
 * 1. Top Left: nw, n, w, m
 * 2. Top: n, m
 * 3. Top Right: ne, n, e, m
 * 4. Left: w, m
 * 5. Middle: m
 * 6. Right: e, m
 * 7. Bottom Left: sw, s, w, m
 * 8. Bottom: s, m
 * 9. Bottom Right: se, s, e, m
 */
class GaussianBlurCodeletHalf4 : public Vertex {

public:
    Input <Vector<half, VectorLayout::ONE_PTR, 8>> in;
    Input <Vector<half, VectorLayout::ONE_PTR, 8>> nw, ne, sw, se;
    Input <Vector<half, VectorLayout::ONE_PTR, 8>> n, s, w, e;
    Output <Vector<half, VectorLayout::ONE_PTR, 8>> out;
    unsigned width;
    unsigned height;

    bool compute() {
        const auto nx = width;
        const auto ny = height;

        const auto h4in = AS_H4(in);
        auto h4out = AS_H4(out);

//         Only works if this is at least a 2x2 block (excluding halos), and in must be same size as out
        if (nx > 1 && ny > 1) {
            // top left
            {
                constexpr auto x = 0u;
                constexpr auto y = 0u;
                const auto _nw = AS_H4(nw)[0];
                const auto _w = AS_H4(w)[H4_HALO_OFFSET_V(0)];
                const auto _sw = AS_H4(w)[H4_HALO_OFFSET_V(1)];
                const auto _n = AS_H4(n)[H4_HALO_OFFSET_H(0)];
                const auto _m = h4in[H4_IDX(0, 0)];
                const auto _s = h4in[H4_IDX(1, 0)];
                const auto _ne = AS_H4(n)[H4_HALO_OFFSET_H(1)];
                const auto _e = h4in[H4_IDX(0, 1)];
                const auto _se = h4in[H4_IDX(1, 1)];
                h4out[H4_IDX(0, 0)] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);

            }

            // top
            {
                constexpr auto y = 0u;
                for (auto x = 1u; x < nx - 1; x++) {
                    const auto _nw = AS_H4(n)[H4_HALO_OFFSET_H(-1)];
                    const auto _w = h4in[H4_IDX(0, -1)];
                    const auto _sw = h4in[H4_IDX(1, -1)];
                    const auto _n = AS_H4(n)[H4_HALO_OFFSET_H(0)];
                    const auto _m = h4in[H4_IDX(0, 0)];
                    const auto _s = h4in[H4_IDX(1, 0)];
                    const auto _ne = AS_H4(n)[H4_HALO_OFFSET_H(1)];
                    const auto _e = h4in[H4_IDX(0, 1)];
                    const auto _se = h4in[H4_IDX(1, 1)];
                    h4out[H4_IDX(0, 0)] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
                }
            }

            // top right
            {
                const auto x = nx - 1u;
                constexpr auto y = 0u;
                const auto _nw = AS_H4(n)[H4_HALO_OFFSET_H(-1)];
                const auto _w = h4in[H4_IDX(0, -1)];
                const auto _sw = h4in[H4_IDX(1, -1)];
                const auto _n = AS_H4(n)[H4_HALO_OFFSET_H(0)];
                const auto _m = h4in[H4_IDX(0, 0)];
                const auto _s = h4in[H4_IDX(1, 0)];
                const auto _ne = AS_H4(ne)[0];
                const auto _e = AS_H4(e)[H4_HALO_OFFSET_V(0)];
                const auto _se = AS_H4(e)[H4_HALO_OFFSET_V(1)];
                h4out[H4_IDX(0, 0)] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
            }

            // left col
            {
                constexpr auto x = 0u;
                for (auto y = 1; y < ny - 1; y++) {
                    const auto _nw = AS_H4(w)[H4_HALO_OFFSET_V(-1)];
                    const auto _w = AS_H4(w)[H4_HALO_OFFSET_V(0)];
                    const auto _sw = AS_H4(w)[H4_HALO_OFFSET_V(1)];
                    const auto _n = h4in[H4_IDX(-1, 0)];
                    const auto _m = h4in[H4_IDX(0, 0)];
                    const auto _s = h4in[H4_IDX(1, 0)];
                    const auto _ne = h4in[H4_IDX(-1, 1)];
                    const auto _e = h4in[H4_IDX(0, 1)];
                    const auto _se = h4in[H4_IDX(1, 1)];
                    h4out[H4_IDX(0, 0)] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);

                }
            }

            // middle block
            for (auto y = 1; y < ny - 1; y++) {
                for (auto x = 1; x < nx - 1; x++) {
                    const auto _nw = h4in[H4_IDX(-1, -1)];
                    const auto _w = h4in[H4_IDX(0, -1)];
                    const auto _sw = h4in[H4_IDX(+1, -1)];
                    const auto _n = h4in[H4_IDX(-1, 0)];
                    const auto _m = h4in[H4_IDX(0, 0)];
                    const auto _s = h4in[H4_IDX(+1, 0)];
                    const auto _ne = h4in[H4_IDX(+1, +1)];
                    const auto _e = h4in[H4_IDX(0, +1)];
                    const auto _se = h4in[H4_IDX(-1, +1)];
                    h4out[H4_IDX(0, 0)] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);

                }
            }

            // right col
            {
                const auto x = nx - 1u;
                for (auto y = 1; y < ny - 1u; y++) {

                    const auto _nw = h4in[H4_IDX(-1, -1)];
                    const auto _w = h4in[H4_IDX(0, -1)];
                    const auto _sw = h4in[H4_IDX(1, -1)];
                    const auto _n = h4in[H4_IDX(-1, 0)];
                    const auto _m = h4in[H4_IDX(0, 0)];
                    const auto _s = h4in[H4_IDX(1, 0)];
                    const auto _ne = AS_H4(e)[H4_HALO_OFFSET_V(-1)];
                    const auto _e = AS_H4(e)[H4_HALO_OFFSET_V(0)];
                    const auto _se = AS_H4(e)[H4_HALO_OFFSET_V(1)];
                    h4out[H4_IDX(0, 0)] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
                }
            }

            // bottom left
            {
                const auto y = ny - 1;
                constexpr auto x = 0u;

                const auto _nw = AS_H4(w)[H4_HALO_OFFSET_V(-1)];
                const auto _w = AS_H4(w)[H4_HALO_OFFSET_V(0)];
                const auto _sw = AS_H4(sw)[0];
                const auto _n = h4in[H4_IDX(-1, 0)];
                const auto _m = h4in[H4_IDX(0, 0)];
                const auto _s = AS_H4(s)[H4_HALO_OFFSET_H(0)];
                const auto _ne = h4in[H4_IDX(-1, 1)];
                const auto _e = h4in[H4_IDX(0, 1)];
                const auto _se = AS_H4(s)[H4_HALO_OFFSET_H(1)];
                h4out[H4_IDX(0, 0)] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
            }

            // bottom
            {
                const auto y = ny - 1;
                for (auto x = 1u; x < nx - 1u; x++) {
                    const auto _nw = h4in[H4_IDX(-1, -1)];
                    const auto _w = h4in[H4_IDX(0, -1)];
                    const auto _sw = AS_H4(s)[H4_HALO_OFFSET_H(-1)];
                    const auto _n = h4in[H4_IDX(-1, 0)];
                    const auto _m = h4in[H4_IDX(0, 0)];
                    const auto _s = AS_H4(s)[H4_HALO_OFFSET_H(0)];
                    const auto _ne = h4in[H4_IDX(-1, 1)];
                    const auto _e = h4in[H4_IDX(0, 1)];
                    const auto _se = AS_H4(s)[H4_HALO_OFFSET_H(1)];
                    h4out[H4_IDX(0, 0)] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
                }
            }

            // bottom right
            {
                const auto y = ny - 1;
                const auto x = nx - 1;

                const auto _nw = h4in[H4_IDX(-1, -1)];
                const auto _w = h4in[H4_IDX(0, -1)];
                const auto _sw = AS_H4(s)[H4_HALO_OFFSET_H(-1)];
                const auto _n = h4in[H4_IDX(-1, 0)];
                const auto _m = h4in[H4_IDX(0, 0)];
                const auto _s = AS_H4(s)[H4_HALO_OFFSET_H(0)];
                const auto _ne = AS_H4(e)[H4_HALO_OFFSET_V(-1)];
                const auto _e = AS_H4(e)[H4_HALO_OFFSET_V(0)];
                const auto _se = AS_H4(se)[0];
                h4out[H4_IDX(0, 0)] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
            }
            return true;
        }
        return false;
    }
};


/**
 * The half4 vectorised version of the special case where the worker's partition 1 row high, but >1 cells wide.
 * We use the vectorised stencil to do the RG channels at the same time, then the BA channels
 * Now we have 3 special loop cases which have to take into account that cell neighbours may be from different
 * data structures (the halos). The
 * regions are: (as region name: [data structures used]):
 * 1. Left: nw, n, w, sw, s, m
 * 2. Middle: n, m, s
 * 3. Right: ne, n, e, se, s, m
 */
class GaussianWide1RowBlurCodeletHalf4 : public Vertex {

public:
    Input <Vector<half, VectorLayout::ONE_PTR, 8>> in;
    Input <Vector<half, VectorLayout::ONE_PTR, 8>> nw, ne, sw, se;
    Input <Vector<half, VectorLayout::ONE_PTR, 8>> n, s, w, e;
    Output <Vector<half, VectorLayout::ONE_PTR, 8>> out;
    unsigned width;
    unsigned height; // Unused, but we want to keep the interface the same

    bool compute() {
        const auto h4in = AS_H4(in);
        auto h4out = AS_H4(out);
        const auto nx = width;
        constexpr auto ny = 1;
        // Only works if this is at least a 1x2 block (excluding halos), and in must be same size as out
        if (height == 1 && nx > 1) {
            constexpr auto y = 0u;
            // Left
            {
                const auto x = 0u;
                const auto _nw = AS_H4(nw)[0];
                const auto _w = AS_H4(w)[0];
                const auto _sw = AS_H4(sw)[0];
                const auto _n = AS_H4(n)[H4_HALO_OFFSET_H(0)];
                const auto _m = h4in[H4_IDX(0, 0)];
                const auto _s = AS_H4(s)[H4_HALO_OFFSET_H(0)];
                const auto _ne = AS_H4(n)[H4_HALO_OFFSET_H(1)];
                const auto _e = h4in[H4_IDX(0, 1)];
                const auto _se = AS_H4(s)[H4_HALO_OFFSET_H(1)];
                h4out[H4_IDX(0, 0)] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
            }

            // Middle
            for (auto x = 1u; x < nx - 1; x++) {
                const auto _nw = AS_H4(n)[H4_HALO_OFFSET_H(-1)];
                const auto _w = h4in[H4_IDX(0, -1)];
                const auto _sw = AS_H4(s)[H4_HALO_OFFSET_H(-1)];
                const auto _n = AS_H4(n)[H4_HALO_OFFSET_H(0)];
                const auto _m = h4in[H4_IDX(0, 0)];
                const auto _s = AS_H4(s)[H4_HALO_OFFSET_H(0)];
                const auto _ne = AS_H4(n)[H4_HALO_OFFSET_H(1)];
                const auto _e = h4in[H4_IDX(0, 1)];
                const auto _se = AS_H4(s)[H4_HALO_OFFSET_H(1)];
                h4out[H4_IDX(0, 0)] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
            }

            // Right
            {
                const auto x = nx - 1u;
                const auto _nw = AS_H4(n)[H4_HALO_OFFSET_H(-1)];
                const auto _w = h4in[H4_IDX(0, -1)];
                const auto _sw = AS_H4(s)[H4_HALO_OFFSET_H(-1)];
                const auto _n = AS_H4(n)[H4_HALO_OFFSET_H(0)];
                const auto _m = h4in[H4_IDX(0, 0)];
                const auto _s = AS_H4(s)[H4_HALO_OFFSET_H(0)];
                const auto _ne = AS_H4(n)[H4_HALO_OFFSET_H(1)];
                const auto _e = h4in[H4_IDX(0, 1)];
                const auto _se = AS_H4(s)[H4_HALO_OFFSET_H(1)];
                h4out[H4_IDX(0, 0)] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
            }
            return true;
        }
        return false;
    }

};


/**
 * The half4 vectorised version of the special case where the worker's partition 1 column wide, but >1 cells high. Now we have 3 special loop cases
 * which have to take into account that cell neighbours may be from different data structures (the halos). We do the RG, then BA channels at the same time.
 * The regions are: (as region name: [data structures used]):
 * 1. Top: nw, n, ne, w, e, m
 * 2. Middle: w, m, e
 * 3. Bottom: sw, w, m, se, s, e
 */
template<typename T>
class GaussianNarrow1ColBlurCodeletHalf4 : public Vertex {

public:
    Input <Vector<half, VectorLayout::ONE_PTR, 8>> in;
    Input <Vector<half, VectorLayout::ONE_PTR, 8>> nw, ne, sw, se;
    Input <Vector<half, VectorLayout::ONE_PTR, 8>> n, s, w, e;
    Output <Vector<half, VectorLayout::ONE_PTR, 8>> out;
    unsigned width;
    unsigned height; // Unused, but we want to keep the interface the same

    bool compute() {
        const auto ny = height;
        constexpr auto nx = 1;
        const auto h4in = AS_H4(in);
        auto h4out = AS_H4(out);

        // Only works if this is at least a 2x1 block (excluding halos), and in must be same size as out
        if (width == 1 && ny > 1) {
            constexpr auto x = 0u;
            // Top
            {
                constexpr auto y = 0u;

                const auto _nw = AS_H4(nw)[0];
                const auto _w = AS_H4(w)[H4_HALO_OFFSET_V(0)];
                const auto _sw = AS_H4(w)[H4_HALO_OFFSET_V(1)];
                const auto _n = AS_H4(n)[0];
                const auto _m = h4in[H4_IDX(0, 0)];
                const auto _s = h4in[H4_IDX(1, 0)];
                const auto _ne = AS_H4(ne)[0];
                const auto _e = AS_H4(e)[H4_HALO_OFFSET_V(0)];
                const auto _se = AS_H4(e)[H4_HALO_OFFSET_V(1)];
                h4out[H4_IDX(0, 0)] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
            }

            // Middle
            {
                for (auto y = 1u; y < ny - 1; y++) {
                    const auto _nw = AS_H4(w)[H4_HALO_OFFSET_V(-1)];
                    const auto _w = AS_H4(w)[H4_HALO_OFFSET_V(0)];
                    const auto _sw = AS_H4(w)[H4_HALO_OFFSET_V(1)];
                    const auto _n = h4in[H4_IDX(-1, 0)];
                    const auto _m = h4in[H4_IDX(0, 0)];
                    const auto _s = h4in[H4_IDX(1, 0)];
                    const auto _ne = AS_H4(e)[H4_HALO_OFFSET_V(-1)];
                    const auto _e = AS_H4(e)[H4_HALO_OFFSET_V(0)];
                    const auto _se = AS_H4(e)[H4_HALO_OFFSET_V(1)];
                    h4out[H4_IDX(0, 0)] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
                }
            }

            // Right
            {
                const auto y = ny - 1u;

                const auto _nw = AS_H4(w)[H4_HALO_OFFSET_V(-1)];
                const auto _w = AS_H4(w)[H4_HALO_OFFSET_V(0)];
                const auto _sw = AS_H4(sw)[0];
                const auto _n = h4in[H4_IDX(-1, 0)];
                const auto _m = h4in[H4_IDX(0, 0)];
                const auto _s = AS_H4(s)[0];
                const auto _ne = AS_H4(e)[H4_HALO_OFFSET_V(-1)];
                const auto _e = AS_H4(e)[H4_HALO_OFFSET_V(0)];
                const auto _se = AS_H4(se)[0];
                h4out[H4_IDX(0, 0)] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
            }

            return true;
        }
        return false;
    }
};


/**
 * Vectorised half4 version of the extreme case, where the worker's partition 1 column wide and 1 cell high. Now all neighbours come from the
 * 8 halo regions. We do the RG channels together then the BA channels
 */
template<typename T>
class GaussianBlur1x1CodeletHalf4 : public Vertex { // Extreme case of a 1x1 partition!

public:
    Input <Vector<half, VectorLayout::ONE_PTR, 8>> in;
    Input <Vector<half, VectorLayout::ONE_PTR, 8>> nw, ne, sw, se;
    Input <Vector<half, VectorLayout::ONE_PTR, 8>> n, s, w, e;
    Output <Vector<half, VectorLayout::ONE_PTR, 8>> out;
    unsigned width;
    unsigned height; // Unused, but we want to keep the interface the same for easy re-use

    bool compute() {
        const auto ny = height;
        const auto nx = width;

        // Only works if this is at least a 1x1 block (excluding halos), and in must be same size as out
        if (nx == 1 && ny > 1) {
            constexpr auto x = 0u;
            constexpr auto y = 0u;
            const auto h4in = AS_H4(in);
            auto h4out = AS_H4(out);

            const auto _nw = AS_H4(nw)[0];
            const auto _w = AS_H4(w)[0];
            const auto _sw = AS_H4(sw)[0];
            const auto _n = AS_H4(n)[0];
            const auto _m = h4in[0];
            const auto _s = AS_H4(s)[0];
            const auto _ne = AS_H4(ne)[0];
            const auto _e = AS_H4(e)[0];
            const auto _se = AS_H4(se)[0];

            h4out[H4_IDX(0, 0)] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);

            return true;
        }
        return false;
    }
};
