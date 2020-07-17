#include <poplar/Vertex.hpp>
#include <cstddef>
#include <print.h>
#include <math.h>
#include <ipudef.h>


using namespace poplar;
auto constexpr NumChannels = 4;

class GaussianBlurCodelet : public Vertex {

public:
    Input <Vector<float>> haloTop; // numCols x NumChans
    Input <Vector<float>> haloLeft; // numRows x NumChans
    Input <Vector<float>> haloRight; // numRows x NumChans
    Input <Vector<float>> haloBottom; // numCols x NumChans
    Input<float> haloTopLeft; // NumChans
    Input<float> haloTopRight;// NumChans
    Input<float> haloBottomLeft;// NumChans
    Input<float> haloBottomRight;// NumChans

    Input <Vector<float>> in; // numCols x numRows x NumChans
    Input<unsigned> numRows; // i.e. excluding halo
    Input<unsigned> numCols; // i.e. excluding halo

    Output <Vector<float>> out; // numCols x numRows x NumChans

    bool compute() {
        const auto nc = *numCols;
        const auto nr = *numRows;

        // Remember layout is (0,0) = top left
        const auto TOP = 0ul;
        constexpr auto BOTTOM = nr - 1;
        constexpr auto LEFT = 0ul;
        const auto RIGHT = nc - 1;
        const int northCellOffset = +((int) nc);
        const int southCellOffset = -((int) nc);
        constexpr int eastCellOffset = +(int) 1;
        constexpr int westCellOffset = -(int) 1;
        constexpr int middleCellOffset = 0;
        constexpr int sideHaloNorthCellOffset = +(int) 1; // when we are using the left or right halo and want to know what up is
        constexpr int sideHaloSouthCellOffset = -(int) 1;// when we are using the left or right halo and want to know what down is
        const int northWestCellOffset = +northCellOffset + westCellOffset;
        const int northEastCellOffset = +northCellOffset + eastCellOffset;
        const int southEastCellOffset = +southCellOffset + eastCellOffset;
        const int southWestCellOffset = +southCellOffset + westCellOffset;


        for (auto row = BOTTOM; row <= TOP; row++) {
            for (auto col = LEFT; col <= RIGHT; col++) {
                for (auto chan = 0; chan < NumChannels; chan++) {
                    const auto offset = [chan](size_t i) -> size_t {return i * NumChannels + chan;};

                    const int cellIdx = row * nc + col;
                    const auto n = [=]() -> float {
                        if (row == TOP) {
                            return haloTop[offset(col + middleCellOffset)];
                        }
                        return in[offset(cellIdx + northCellOffset)];
                    }();

                    const auto ne = [=]() -> float {
                        if (row == TOP && col == RIGHT) {
                            return haloTopRight[offset(0)];
                        } else if (row == TOP) {
                            return haloTop[offset(col + eastCellOffset)];
                        } else if (col == RIGHT) {
                            return haloRight[offset(row + sideHaloNorthCellOffset)];
                        }
                        return in[offset(cellIdx + northEastCellOffset)];
                    }();

                    const auto nw = [=]() -> float {
                        if (row == TOP && col == LEFT) {
                            return haloTopLeft[offset(0)];
                        } else if (row == TOP) {
                            return haloTop[offset(col + westCellOffset)];
                        } else if (col == LEFT) {
                            return haloLeft[offset(row + sideHaloNorthCellOffset)];
                        }
                        return in[cellIdx + northWestCellOffset];
                    }();

                    const auto w = [=]() -> float {
                        if (col == LEFT) {
                            return haloLeft[offset(row + middleCellOffset)];
                        }
                        return in[offset(cellIdx + westCellOffset)];
                    }();

                    const auto m = [=]() -> float {
                        return in[offset(cellIdx + middleCellOffset)];
                    }();

                    const auto e = [=]() -> float {
                        if (col == RIGHT) {
                            return haloRight[offset(row + middleCellOffset)];
                        }
                        return in[offset(cellIdx + eastCellOffset)];
                    }();

                    const auto s = [=]() -> float {
                        if (row == BOTTOM) {
                            return haloBottom[offset(col + middleCellOffset)];
                        }
                        return in[offset(cellIdx + southCellOffset)];
                    }();

                    const auto se = [=]() -> float {
                        if (row == BOTTOM && col == RIGHT) {
                            return haloBottomRight[offset(0)];
                        } else if (row == BOTTOM) {
                            return haloBottom[offset(col + eastCellOffset)];
                        } else if (col == RIGHT) {
                            return haloRight[offset(row + sideHaloSouthCellOffset)];
                        }
                        return in[offset(cellIdx + southEastCellOffset)];
                    }();


                    const auto sw = [=]() -> float {
                        if (row == BOTTOM && col == LEFT) {
                            return haloBottomLeft[offset(0)];
                        } else if (row == BOTTOM) {
                            return haloBottom[offset(col + westCellOffset)];
                        } else if (col == LEFT) {
                            return haloLeft[offset(row + sideHaloSouthCellOffset)];
                        }
                        return in[offset(cellIdx + southWestCellOffset)];
                    }();

                    out[offset(cellIdx)] = ((ne + nw + se + sw) + 2.f * (n + s + e + w) + 4.f * m) / 16.f;
                }
            }
        }
        return true;
    }
};
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
