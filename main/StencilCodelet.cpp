#include <poplar/Vertex.hpp>
#include <cstddef>
#include <print.h>
#include <math.h>
#include <ipudef.h>


using namespace poplar;
auto constexpr NumChannels = 4;
using namespace poplar;

template<typename T>
T stencil(const T nw, const T n, const T ne, const T w, const T m,
          const T e, const T sw,
          const T s, const T se) {
    return 1.f / 16 * (nw + ne + sw + se) + 4.f / 16 * m + 2.f / 16 * (e + w + s + n);
}

template<typename T>
class GaussianBlurCodelet : public Vertex {

public:
    Input <Vector<T>> in;
    Input <Vector<T>> nw, ne, sw, se;
    Input <Vector<T>> n, s, w, e;
    Output <Vector<T>> out;
    Input<unsigned> width;
    Input<unsigned> height;

    bool compute() {
        const auto nx = *width;
        const auto ny = *height;
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
                    const auto _sw = w[NumChannels * (nx * (y + 0) + (x + 0)) + c];
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


template<typename T>
class GaussianWide1RowBlurCodelet : public Vertex {

public:
    Input <Vector<T>> in;
    Input <Vector<T>> nw, ne, sw, se;
    Input <Vector<T>> n, s, w, e;
    Output <Vector<T>> out;
    Input<unsigned> width;
    Input<unsigned> height; // Unused, but we want to keep the interface the same

    bool compute() {
        const auto nx = *width;
        constexpr auto nc = NumChannels;

//         Only works if this is at least a 1x2 block (excluding halos), and in must be same size as out
        if (*height == 1 && nx > 1) {
            // Left
            {
#pragma unroll 4
                for (auto c = 0u; c < nc; c++) {
                    const auto lIdx = c;
                    const auto mIdx = c;
                    const auto rIdx = NumChannels + 1;
                    const auto _nw = nw[lIdx];
                    const auto _w = w[lIdx];
                    const auto _sw = sw[lIdx];
                    const auto _n = n[mIdx];
                    const auto _m = in[mIdx];
                    const auto _s = s[mIdx];
                    const auto _ne = n[rIdx];
                    const auto _e = in[rIdx];
                    const auto _se = s[rIdx];
                    out[mIdx] = stencil(_nw, _n, _ne, _w, _m, _e, _sw, _s, _se);
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

template<typename T>
class GaussianNarrow1ColBlurCodelet : public Vertex {

public:
    Input <Vector<T>> in;
    Input <Vector<T>> nw, ne, sw, se;
    Input <Vector<T>> n, s, w, e;
    Output <Vector<T>> out;
    Input<unsigned> width;
    Input<unsigned> height; // Unused, but we want to keep the interface the same

    bool compute() {
        const auto ny = *height;
        constexpr auto nc = NumChannels;

//         Only works if this is at least a 1x2 block (excluding halos), and in must be same size as out
        if (*width == 1 && ny > 1) {
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



//
//
//class GaussianBlurCodeletUnrolled : public Vertex {
//
//public:
//    Input <Vector<float>> haloTop; // numCols
//    Input <Vector<float>> haloLeft; // numRows
//    Input <Vector<float>> haloRight; // numRows
//    Input <Vector<float>> haloBottom; // numCols
//    Input<float> haloTopLeft;
//    Input<float> haloTopRight;
//    Input<float> haloBottomLeft;
//    Input<float> haloBottomRight;
//
//    Input <Vector<float>> in; // numCols x numRows
//    Input<unsigned> numRows; // i.e. excluding halo
//    Input<unsigned> numCols; // i.e. excluding halo
//
//    Output <Vector<float>> out; // numCols x numRows
//
//    bool compute() {
//        const auto nc = *numCols;
//        const auto nr = *numRows;
//        // Contract: this codelet is for grids only (not 1-cell, 1-row or 1-col situations)
//        if (nc <= 1 && nr <= 1) return false;
//
//        const auto kernel = [](const float nw, const float n, const float ne,
//                               const float w, const float m, const float e, const float sw, const float s,
//                               const float se) -> {
//            return ((ne + nw + se + sw) + 2.f * (n + s + e + w) + 4.f * m) / 16.f;
//        };
//
//        // Remember layout is (0,0) = top left
//        const auto TOP = 0ul;
//        constexpr auto BOTTOM = nr - 1;
//        constexpr auto LEFT = 0ul;
//        const auto RIGHT = nc - 1;
//        const int northCellOffset = +((int) nc);
//        const int southCellOffset = -((int) nc);
//        constexpr int eastCellOffset = +(int) 1;
//        constexpr int westCellOffset = -(int) 1;
//        constexpr int middleCellOffset = 0;
//        constexpr int sideHaloNorthCellOffset = +(int) 1; // when we are using the left or right halo and want to know what up is
//        constexpr int sideHaloSouthCellOffset = -(int) 1;// when we are using the left or right halo and want to know what down is
//        const int northWestCellOffset = +northCellOffset + westCellOffset;
//        const int northEastCellOffset = +northCellOffset + eastCellOffset;
//        const int southEastCellOffset = +southCellOffset + eastCellOffset;
//        const int southWestCellOffset = +southCellOffset + westCellOffset;
//
//
//        const auto topRowNorthCell = [=](size_t row, size_t col) -> { return haloTop[col + middleCellOffset]; };
//        const auto otherNorthCell = [=](size_t row, size_t col) -> { return in[cellIdx + northCellOffset]; };
//
//        const auto topRightNorthEastCell = [=](size_t row, size_t col) -> { return *haloTopRight; };
//        const auto topRowNorthEastCell = [=](size_t row, size_t col) -> { return haloTop[col + eastCellOffset]; };
//        const auto rightColNorthEastCell = [=](size_t row, size_t col) -> {
//            return haloRight[row + sideHaloNorthCellOffset];
//        };
//        const auto otherNorthEastCell = [=](size_t row, size_t col) -> { return in[cellIdx + northEastCellOffset]; };
//
//        const auto topLeftNorthWestCell = [=](size_t row, size_t col) -> { return *haloTopLeft; };
//        const auto topRowNorthWestCell = [=](size_t row, size_t col) -> { return haloTop[col + westCellOffset]; };
//        const auto leftColNorthWestCell = [=](size_t row, size_t col) -> {
//            return haloLeft[row + sideHaloNorthCellOffset];
//        };
//        const auto otherNorthWestCell = [=](size_t row, size_t col) -> { return in[cellIdx + northWestCellOffset]; };
//
//        const auto leftColWestCell = [=](size_t row, size_t col) -> { return haloLeft[row + middleCellOffset]; };
//        const auto otherWestCell = [=](size_t row, size_t col) -> { return in[cellIdx + westCellOffset]; };
//
//        const auto middleCell = [=](size_t row, size_t col) -> { return in[cellIdx + middleCellOffset]; };
//
//        const auto rightColEastCell = [=](size_t row, size_t col) -> { return haloRight[row + middleCellOffset]; };
//        const auto otherRightCell = [=](size_t row, size_t col) -> { return in[cellIdx + eastCellOffset]; };
//
//        const auto bottomRowSouthCell = [=](size_t row, size_t col) -> { return haloBottom[col + middleCellOffset]; };
//        const auto otherSouthCell = [=](size_t row, size_t col) -> { return in[cellIdx + southCellOffset]; };
//
//        const auto bottomRightSouthEastCell = [=](size_t row, size_t col) -> { return *haloBottomRight; };
//        const auto bottomRowSouthEastCell = [=](size_t row, size_t col) -> { return haloBottom[col + eastCellOffset]; };
//        const auto rightRowSouthEastCell = [=](size_t row, size_t col) -> {
//            return haloRight[row + sideHaloSouthCellOffset];
//        };
//        const auto otherSouthEastCell = [=](size_t row, size_t col) -> { return in[cellIdx + southEastCellOffset]; };
//
//
//        const auto bottomLeftSouthWestCell = [=](size_t row, size_t col) -> { return *haloBottomLeft; };
//        const auto bottomSouthWestCell = [=](size_t row, size_t col) -> { return haloBottom[col + westCellOffset]; };
//        const auto leftSouthWestCell = [=](size_t row, size_t col) -> {
//            return haloLeft[row + sideHaloSouthCellOffset];
//        };
//        const auto otherSouthWestCell = [=](size_t row, size_t col) -> { return in[cellIdx + southWestCellOffset]; };
//
//
//        const auto
//        applyKernelToRange(const auto r_from, const auto r_to, const auto c_from, const auto c_to, const auto f) {
//            for (auto row = r_from; row <= r_to; row++) {
//                for (auto col = c_from; col <= c_to; col++) {
//                    const int cellIdx = row * nc + col;
//                    out[cellIdx] = f();
//                }
//            }
//        }
//
//        // TL cell
//        kernel(
//                topLeftNorthWestCell,
//                topRowNorthCell,
//                topRowNorthEastCell,
//                leftColWestCell,
//                middleCell,
//                otherRightCell,
//                leftSouthWestCell,
//                otherSouthCell,
//                otherSouthEastCell)
//        );
//        // Top row cell
//        applyKernelToRange(0, 1, 1, nc - 1, kernel(
//                topRowNorthWestCell,
//                topRowNorthCell,
//                topRowNorthEastCell))
//
//        return true;
//    }
//
//};
//
//
//class GaussianBlurOneRowCodeletUnrolled : public Vertex {
//
//public:
//    Input <Vector<float, poplar::VectorLayout::PTR_ONLY>> top; // numCols
//    Input <Vector<float>> bottom; // numCols
//    Input <Vector<float>> left; // numCols
//    Input <Vector<float>> right; // numCols
//
//    Input <Vector<float>> in; // numCols x numRows
//    Input<unsigned> numCols; // i.e. excluding halo
//
//    Output <Vector<float>> out; // numCols x numRows
//
//    bool compute() {
//        const auto RIGHT = *numRows - 1;
//
//        auto x10 = *haloLeft;
//        auto x11 = in[0];
//
//        auto s0020 = top[1] + bottom[1];
//        auto s0121 = top[0] + bottom[0];
//        auto s0222 = 0;
//        auto x12 = 0;
//
//        auto col = 0;
//
//        while (true) {
//
//            if (col == RIGHT) { // special case: about to do last col
//                x12 = *haloRight;
//                s0222 = *haloTopRight + *haloBottomRight;
//                out[col] = ((s0020 + s0222) + 2.f * (s0121 + x10 + x12) + 4.f * x11) / 16.f;
//                break;
//            }
//            x12 = out[col + 1];
//            s0222 = haloTop[col + 1] + haloBottom[col + 1];
//            out[col] = ((s0020 + s0222) + 2.f * (s0121 + x10 + x12) + 4.f * x11) / 16.f;
//
//            if (col + 1 == RIGHT) { // special case: last col
//                x10 = *haloRight;
//                s0220 = *haloTopRight + *haloBottomRight;
//                out[col + 1] = ((s0121 + s0020) + 2.f * (s0212 + x11 + x10) + 4.f * x12) / 16.f;
//                break;
//            }
//            x10 = e;
//            s0220 = ne + se;
//            out[col + 1] = ((s0121 + s0020) + 2.f * (s0212 + x11 + x10) + 4.f * x12) / 16.f;
//
//            if (col + 2 == RIGHT) { // special case: last col
//                x11 = *haloRight;
//                s1221 = *haloTopRight + *haloBottomRight;
//                out[col + 2] = ((s0222 + s0121) + 2.f * (s0020 + x12 + x11) + 4.f * x10) / 16.f;
//                break;
//            }
//            x11 = e;
//            s1221 = ne + se;
//            out[col + 2] = ((s0222 + s0121) + 2.f * (s0020 + x12 + x11) + 4.f * x10) / 16.f;
//
//            col += 2;
//        }
//        return true;
//    }
//};
//
