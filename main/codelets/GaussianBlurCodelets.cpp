#include <poplar/Vertex.hpp>
#include <cstddef>
#include <print.h>
#include <math.h>
#include <ipudef.h>

using namespace poplar;

/**
 * Gaussian Blur on 4 channels (RGBA) of the input image. We have different codelets to deal with different partition
 * sizes. In general, a codelet will deal with a region that is bigger than 1x1 cell, and have to select a cell's
 * neighbours appropriately from the halo data structures it is provided with
 */

//TODO rename bad _nw style identifiers

auto constexpr NumChannels = 4;


// ------------------------------------- Unvectorised implementation -----------------------------------------------
template<typename T>
float stencil(const T nw, const T n, const T ne, const T w, const T m,
          const T e, const T sw,
          const T s, const T se) {
    constexpr auto sixteenth = (T) 1.f / 16;
    constexpr auto quarter = (T) 4.f / 16;
    constexpr auto eighth = (T) 2.f / 16;
    return sixteenth * (nw + ne + sw + se) +  quarter * m + eighth * (e + w + s + n);
}

/**
 * The general case, where the worker's partition is >= 2x2. In this case there are 9 different loop regions,
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
template<typename T>
class GaussianBlurCodelet : public Vertex {

public:
    Input <Vector<T, VectorLayout::ONE_PTR, 8>> in;
    Input <Vector<T, VectorLayout::ONE_PTR, 8>> nw, ne, sw, se;
    Input <Vector<T, VectorLayout::ONE_PTR, 8>> n, s, w, e;
    Output <Vector<T, VectorLayout::ONE_PTR, 8>> out;
    unsigned width;
    unsigned height;

    bool compute() {
        const auto nx = width;
        const auto ny = height;
        const auto nc = 4;

//         Only works if this is at least a 3x3 block (excluding halos), and in must be same size as out
        if (nx > 1 && ny > 1) {
            // top left
            {
                constexpr auto x = 0u;
                constexpr auto y = 0u;
#pragma unroll 4
                for (auto c = 0u; c < nc; c++) {
                    const auto idx = NumChannels * (nx * (y + 0) + (x + 0)) + c;
                    const auto _nw = nw[c];
                    const auto _w = w[c];
                    const auto _sw = w[c + NumChannels];
                    const auto _n = n[c];
                    const auto _m = in[idx];
                    const auto _s = in[NumChannels * (nx * (y + 1) + (x + 0)) + c];
                    const auto _ne = n[c + NumChannels];
                    const auto _e = in[NumChannels * (nx * (y + 0) + (x + 1)) + c];
                    const auto _se = in[NumChannels * (nx * (y + 1) + (x + 1)) + c];
                    out[idx] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
                }
            }

            // top
            {
                constexpr auto y = 0u;
#pragma unroll 2
                for (auto x = 1u; x < nx - 1; x++) {
#pragma unroll 4
                    for (auto c = 0; c < nc; c++) {
                        const auto idx = NumChannels * (nx * (y + 0) + (x + 0)) + c;
                        const auto _nw = n[NumChannels * (x - 1) + c];
                        const auto _w = in[NumChannels * (nx * (y + 0) + (x - 1)) + c];
                        const auto _sw = in[NumChannels * (nx * (y + 1) + (x - 1)) + c];
                        const auto _n = n[NumChannels * (x + 0) + c];
                        const auto _m = in[idx];
                        const auto _s = in[NumChannels * (nx * (y + 1) + (x + 0)) + c];
                        const auto _ne = n[NumChannels * (x + 1) + c];
                        const auto _e = in[NumChannels * (nx * (y + 0) + (x + 1)) + c];
                        const auto _se = in[NumChannels * (nx * (y + 1) + (x + 1)) + c];
                        out[idx] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
                    }
                }
            }

            // top right
            {
                const auto x = nx - 1u;
                constexpr auto y = 0u;
#pragma unroll 4
                for (auto c = 0; c < nc; c++) {
                    const auto idx = NumChannels * (nx * (y + 0) + (x + 0)) + c;
                    const auto _nw = n[NumChannels * (x - 1) + c];
                    const auto _w = in[NumChannels * (nx * (y + 0) + (x - 1)) + c];
                    const auto _sw = in[NumChannels * (nx * (y + 1) + (x - 1)) + c];
                    const auto _n = n[NumChannels * (x + 0) + c];
                    const auto _m = in[idx];
                    const auto _s = in[NumChannels * (nx * (y + 1) + (x + 0)) + c];
                    const auto _ne = ne[c];
                    const auto _e = e[NumChannels * (y + 0) + c];
                    const auto _se = e[NumChannels * (y + 1) + c];
                    out[idx] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
                }
            }

            // left col
            {
                constexpr auto x = 0u;
#pragma unroll 2
                for (auto y = 1; y < ny - 1; y++) {
#pragma unroll 4
                    for (auto c = 0; c < nc; c++) {
                        const auto idx = NumChannels * (nx * (y + 0) + (x + 0)) + c;
                        const auto _nw = w[NumChannels * (y - 1) + c];
                        const auto _w = w[NumChannels * (y + 0) + c];
                        const auto _sw = w[NumChannels * (y + 1) + c];
                        const auto _n = in[NumChannels * (nx * (y - 1) + (x + 0)) + c];
                        const auto _m = in[idx];
                        const auto _s = in[NumChannels * (nx * (y + 1) + (x + 0)) + c];
                        const auto _ne = in[NumChannels * (nx * (y - 1) + (x + 1)) + c];
                        const auto _e = in[NumChannels * (nx * (y + 0) + (x + 1)) + c];
                        const auto _se = in[NumChannels * (nx * (y + 1) + (x + 1)) + c];
                        out[idx] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
                    }
                }
            }

            // middle block
            for (auto y = 1; y < ny - 1; y++) {
#pragma unroll 2
                for (auto x = 1; x < nx - 1; x++) {
#pragma unroll 4
                    for (auto c = 0; c < nc; c++) {
                        const auto idx = NumChannels * (nx * (y + 0) + (x + 0)) + c;
                        const auto _nw = in[NumChannels * (nx * (y - 1) + (x - 1)) + c];
                        const auto _w = in[NumChannels * (nx * (y + 0) + (x - 1)) + c];
                        const auto _sw = in[NumChannels * (nx * (y + 1) + (x - 1)) + c];
                        const auto _n = in[NumChannels * (nx * (y - 1) + (x + 0)) + c];
                        const auto _m = in[idx];
                        const auto _s = in[NumChannels * (nx * (y + 1) + (x + 0)) + c];
                        const auto _ne = in[NumChannels * (nx * (y - 1) + (x + 1)) + c];
                        const auto _e = in[NumChannels * (nx * (y + 0) + (x + 1)) + c];
                        const auto _se = in[NumChannels * (nx * (y + 1) + (x + 1)) + c];
                        out[idx] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
                    }
                }
            }

            // right col
            {
                const auto x = nx - 1u;
#pragma unroll 2
                for (auto y = 1; y < ny - 1u; y++) {
#pragma unroll 4
                    for (auto c = 0; c < nc; c++) {
                        const auto idx = NumChannels * (nx * (y + 0) + (x + 0)) + c;
                        const auto _nw = in[NumChannels * (nx * (y - 1) + (x - 1)) + c];
                        const auto _w = in[NumChannels * (nx * (y + 0) + (x - 1)) + c];
                        const auto _sw = in[NumChannels * (nx * (y + 1) + (x - 1)) + c];
                        const auto _n = in[NumChannels * (nx * (y - 1) + (x + 0)) + c];
                        const auto _m = in[idx];
                        const auto _s = in[NumChannels * (nx * (y + 1) + (x + 0)) + c];
                        const auto _ne = e[NumChannels * (y - 1) + c];
                        const auto _e = e[NumChannels * (y + 0) + c];
                        const auto _se = e[NumChannels * (y + 1) + c];
                        out[idx] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
                    }
                }
            }

            // bottom left
            {
                const auto y = ny - 1;
                constexpr auto x = 0u;
#pragma unroll 4
                for (auto c = 0; c < nc; c++) {
                    const auto idx = NumChannels * (nx * (y + 0) + (x + 0)) + c;
                    const auto _nw = w[NumChannels * (y - 1) + c];
                    const auto _w = w[NumChannels * (y + 0) + c];
                    const auto _sw = sw[c];
                    const auto _n = in[NumChannels * (nx * (y - 1) + (x + 0)) + c];
                    const auto _m = in[idx];
                    const auto _s = s[NumChannels * (x + 0) + c];
                    const auto _ne = in[NumChannels * (nx * (y - 1) + (x + 1)) + c];
                    const auto _e = in[NumChannels * (nx * (y + 0) + (x + 1)) + c];
                    const auto _se = s[NumChannels * (x + 1) + c];
                    out[idx] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
                }
            }

            // bottom
            {
                const auto y = ny - 1;
#pragma unroll 2
                for (auto x = 1u; x < nx - 1u; x++) {
#pragma unroll 4
                    for (auto c = 0; c < nc; c++) {
                        const auto idx = NumChannels * (nx * (y + 0) + (x + 0)) + c;
                        const auto _nw = in[NumChannels * (nx * (y - 1) + (x - 1)) + c];
                        const auto _w = in[NumChannels * (nx * (y + 0) + (x - 1)) + c];
                        const auto _sw = s[NumChannels * (x - 1) + c];
                        const auto _n = in[NumChannels * (nx * (y - 1) + (x + 0)) + c];
                        const auto _m = in[idx];
                        const auto _s = s[NumChannels * (x + 0) + c];
                        const auto _ne = in[NumChannels * (nx * (y - 1) + (x + 1)) + c];
                        const auto _e = in[NumChannels * (nx * (y + 0) + (x + 1)) + c];
                        const auto _se = s[NumChannels * (x + 1) + c];
                        out[idx] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
                    }
                }
            }

            // bottom right
            {
                const auto y = ny - 1;
                const auto x = nx - 1;
#pragma unroll 4
                for (auto c = 0; c < nc; c++) {
                    const auto idx = NumChannels * (nx * (y + 0) + (x + 0)) + c;
                    const auto _nw = in[NumChannels * (nx * (y - 1) + (x - 1)) + c];
                    const auto _w = in[NumChannels * (nx * (y + 0) + (x - 1)) + c];
                    const auto _sw = s[NumChannels * (x - 1) + c];
                    const auto _n = in[NumChannels * (nx * (y - 1) + (x + 0)) + c];
                    const auto _m = in[idx];
                    const auto _s = s[NumChannels * (x + 0) + c];
                    const auto _ne = e[NumChannels * (y - 1) + c];
                    const auto _e = e[NumChannels * (y + 0) + c];
                    const auto _se = se[c];
                    out[idx] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
                }
            }
            return true;
        }
        return false;
    }
};


template
class GaussianBlurCodelet<float>;


/**
 * The special case, where the worker's partition 1 row high, but >1 cells wide. Now we have 3 special loop cases
 * which have to take into account that cell neighbours may be from different data structures (the halos). The
 * regions are: (as region name: [data structures used]):
 * 1. Left: nw, n, w, sw, s, m
 * 2. Middle: n, m, s
 * 3. Right: ne, n, e, se, s, m
 */
template<typename T>
class GaussianWide1RowBlurCodelet : public Vertex {

public:
    Input <Vector<T, VectorLayout::ONE_PTR, 8>> in;
    Input <Vector<T, VectorLayout::ONE_PTR, 8>> nw, ne, sw, se;
    Input <Vector<T, VectorLayout::ONE_PTR, 8>> n, s, w, e;
    Output <Vector<T, VectorLayout::ONE_PTR, 8>> out;
    unsigned width;
    unsigned height; // Unused, but we want to keep the interface the same

    bool compute() {
        const auto nx = width;
        constexpr auto nc = NumChannels;

//         Only works if this is at least a 1x2 block (excluding halos), and in must be same size as out
        if (height == 1 && nx > 1) {
            // Left
            {
#pragma unroll 4
                for (auto c = 0u; c < nc; c++) {
                    const auto _nw = nw[c];
                    const auto _w = w[c];
                    const auto _sw = sw[c];
                    const auto _n = n[c];
                    const auto _m = in[c];
                    const auto _s = s[c];
                    const auto _ne = n[NumChannels + c];
                    const auto _e = in[NumChannels + c];
                    const auto _se = s[NumChannels + c];
                    out[c] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
                }
            }

            // Middle
            {
                constexpr auto y = 0u;
#pragma unroll 2
                for (auto x = 1u; x < nx - 1; x++) {
#pragma unroll 4
                    for (auto c = 0; c < nc; c++) {
                        const auto lIdx = NumChannels * (x - 1) + c; // Index of the col to the left
                        const auto mIdx = NumChannels * x + c; // Index of the middle col
                        const auto rIdx = NumChannels * (x + 1) + c; // Index of the col to the right
                        const auto _nw = n[lIdx];
                        const auto _w = in[lIdx];
                        const auto _sw = s[lIdx];
                        const auto _n = n[mIdx];
                        const auto _m = in[mIdx];
                        const auto _s = s[mIdx];
                        const auto _ne = n[rIdx];
                        const auto _e = in[rIdx];
                        const auto _se = s[rIdx];

                        out[mIdx] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
                    }
                }
            }

            // Right
            {
                const auto x = nx - 1u;
#pragma unroll 4
                for (auto c = 0; c < nc; c++) {
                    const auto lIdx = NumChannels * (x - 1) + c; // Index of the col to the left
                    const auto mIdx = NumChannels * x + c; // Index of the middle col
                    const auto rIdx = c; // Index of the col to the right
                    const auto _nw = n[lIdx];
                    const auto _w = in[lIdx];
                    const auto _sw = s[lIdx];
                    const auto _n = n[mIdx];
                    const auto _m = in[mIdx];
                    const auto _s = s[mIdx];
                    const auto _ne = ne[rIdx];
                    const auto _e = e[rIdx];
                    const auto _se = se[rIdx];

                    out[mIdx] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
                }
            }
            return true;
        }
        return false;
    }
};

template
class GaussianWide1RowBlurCodelet<float>;


/**
 * The special case, where the worker's partition 1 column wide, but >1 cells high. Now we have 3 special loop cases
 * which have to take into account that cell neighbours may be from different data structures (the halos). The
 * regions are: (as region name: [data structures used]):
 * 1. Top: nw, n, ne, w, e, m
 * 2. Middle: w, m, e
 * 3. Bottom: sw, w, m, se, s, e
 */
template<typename T>
class GaussianNarrow1ColBlurCodelet : public Vertex {

public:
    Input <Vector<T, VectorLayout::ONE_PTR, 8>> in;
    Input <Vector<T, VectorLayout::ONE_PTR, 8>> nw, ne, sw, se;
    Input <Vector<T, VectorLayout::ONE_PTR, 8>> n, s, w, e;
    Output <Vector<T, VectorLayout::ONE_PTR, 8>> out;
    unsigned width;
    unsigned height; // Unused, but we want to keep the interface the same

    bool compute() {
        const auto ny = height;
        constexpr auto nc = NumChannels;

//         Only works if this is at least a 2x1 block (excluding halos), and in must be same size as out
        if (width == 1 && ny > 1) {
            // Top
            {
#pragma unroll 4
                for (auto c = 0u; c < nc; c++) {
                    const auto tIdx = c;
                    const auto mIdx = c;
                    const auto bIdx = c;
                    const auto _nw = nw[tIdx];
                    const auto _w = w[mIdx];
                    const auto _sw = w[bIdx];
                    const auto _n = n[tIdx];
                    const auto _m = in[mIdx];
                    const auto _s = in[bIdx];
                    const auto _ne = ne[tIdx];
                    const auto _e = e[mIdx];
                    const auto _se = e[bIdx];
                    out[mIdx] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
                }
            }

            // Middle
            {
                constexpr auto x = 0u;
#pragma unroll 2
                for (auto y = 1u; y < ny - 1; y++) {
#pragma unroll 4
                    for (auto c = 0; c < nc; c++) {
                        const auto tIdx = NumChannels * (y - 1) + c;
                        const auto mIdx = NumChannels * y + c;
                        const auto bIdx = NumChannels * (y + 1) + c;
                        const auto _nw = w[tIdx];
                        const auto _w = w[mIdx];
                        const auto _sw = w[bIdx];
                        const auto _n = in[tIdx];
                        const auto _m = in[mIdx];
                        const auto _s = in[bIdx];
                        const auto _ne = e[tIdx];
                        const auto _e = e[mIdx];
                        const auto _se = e[bIdx];

                        out[mIdx] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
                    }
                }
            }

            // Right
            {
                const auto y = ny - 1u;
#pragma unroll 4
                for (auto c = 0; c < nc; c++) {
                    const auto tIdx = NumChannels * (y - 1) + c;
                    const auto mIdx = NumChannels * y + c;
                    const auto bIdx = c;
                    const auto _nw = w[tIdx];
                    const auto _w = w[mIdx];
                    const auto _sw = sw[bIdx];
                    const auto _n = in[tIdx];
                    const auto _m = in[mIdx];
                    const auto _s = s[bIdx];
                    const auto _ne = e[tIdx];
                    const auto _e = e[mIdx];
                    const auto _se = se[bIdx];

                    out[mIdx] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
                }
            }
            return true;
        }
        return false;
    }
};

template
class GaussianNarrow1ColBlurCodelet<float>;



/**
 * The extreme case, where the worker's partition 1 column wide and 1 cell high. Now all neighbours come from the
 * 8 halo regions
 */
template<typename T>
class GaussianBlur1x1Codelet : public Vertex { // Extreme case of a 1x1 partition!

public:
    Input <Vector<T, VectorLayout::ONE_PTR, 8>> in;
    Input <Vector<T, VectorLayout::ONE_PTR, 8>> nw, ne, sw, se;
    Input <Vector<T, VectorLayout::ONE_PTR, 8>> n, s, w, e;
    Output <Vector<T, VectorLayout::ONE_PTR, 8>> out;
    unsigned width;
    unsigned height; // Unused, but we want to keep the interface the same

    bool compute() {
        const auto ny = height;
        const auto nx = width;
        constexpr auto nc = NumChannels;

//         Only works if this is at least a 1x1 block (excluding halos), and in must be same size as out
        if (nx == 1 && ny > 1) {
#pragma unroll 4
            for (auto c = 0u; c < nc; c++) {
                const auto _nw = nw[c];
                const auto _w = w[c];
                const auto _sw = sw[c];
                const auto _n = n[c];
                const auto _m = in[c];
                const auto _s = s[c];
                const auto _ne = ne[c];
                const auto _e = e[c];
                const auto _se = se[c];
                out[c] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
            }

            return true;
        }
        return false;
    }
};

template
class GaussianBlur1x1Codelet<float>;


