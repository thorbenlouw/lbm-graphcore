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
    return (nw + n + ne + w + m + e + sw + s + se) / (T) 9.0f;
}

#define IDX(R, C)    ((width-2) * (y + R) + x + C)

template<typename T>
class ScanDown : public Vertex {

public:
    InOut <Vector<T, VectorLayout::ONE_PTR, 8>> dataMiddle; //(only the middle bit!)

    Input <Vector<T, VectorLayout::ONE_PTR, 8>> dataTop;
    Input <Vector<T, VectorLayout::ONE_PTR, 8>> dataBottom;
    Input <Vector<T, VectorLayout::ONE_PTR, 8>> dataLeft;
    Input <Vector<T, VectorLayout::ONE_PTR, 8>> dataRight;

    Input <Vector<T, VectorLayout::ONE_PTR, 8>> nw, ne, sw, se; // Sometimes scalars but will be vectors for AoS cases
    Input <Vector<T, VectorLayout::ONE_PTR, 8>> n, s, w, e;

    Output <Vector<T, VectorLayout::ONE_PTR, 8>> borderLeft;
    Output <Vector<T, VectorLayout::ONE_PTR, 8>> borderRight;
    Output <Vector<T, VectorLayout::ONE_PTR, 8>> borderTop;
    Output <Vector<T, VectorLayout::ONE_PTR, 8>> borderBottom;
    unsigned width;
    unsigned height;

    bool compute() {
        // Too many special cases when not 4x4 :-( Just force 4x4 restriction
        // special case! only one row high and one col wide (write to all outputs!)
        // special case: only one row high (write to tmp and top & bottom border)
        // special case: only one col wide (write to left and right borders, and

        // top row special case; write to bottom border (and we have to deal with nw, n and ne here)
        // top-left then middle then top-right
        {
            borderBottom[0] = stencil(nw[0], n[0], n[1],
                                      w[0], dataTop[0], dataTop[1],
                                      w[1], dataMiddle[0], dataMiddle[1]);
            for (auto x = 1u; x < width - 1; x++) {
                borderBottom[x] = stencil(n[x - 1], n[x], n[x + 1],
                                          dataTop[x - 1], dataTop[x], dataTop[x + 1],
                                          dataMiddle[x - 1], dataMiddle[x], dataMiddle[x + 1]);
            }
            borderBottom[width - 1] = stencil(n[width - 2], n[width], ne[0],
                                              dataTop[width - 2], dataTop[width - 1], e[0],
                                              dataMiddle[width - 2], dataMiddle[width - 1], e[1]);
        }

        // row 1 (has to write to borderTop)
        {
            borderTop[0] = stencil(w[0], dataTop[0], dataTop[1],
                                   w[1], dataMiddle[0], dataMiddle[1],
                                   w[2], dataMiddle[width], dataMiddle[width + 1]);
            for (auto x = 1u; x < width - 1; x++) {
                borderTop[x] = stencil(dataTop[x - 1], dataTop[x], dataTop[x + 1],
                                       dataMiddle[x - 1], dataMiddle[x], dataMiddle[x + 1],
                                       dataMiddle[width + x - 1], dataMiddle[width + x], dataMiddle[width + x + 1]);
            }
            borderTop[width - 1] = stencil(dataTop[width - 2], dataTop[width - 1], e[0],
                                           dataMiddle[width - 2], dataMiddle[width - 1], e[1],
                                           dataMiddle[width * 2 - 2], dataMiddle[width * 2 - 1], e[2]);
        }
        // middle:
        //    do left then middle then right
        // cases ne n nw
        // cases w m e
        // sw s se
        for (auto y = 2u; y < height - 1; y++) {

            {
                constexpr x = 0u;
                borderLeft[y - 2] = stencil(w[y - 1], dataMiddle[IDX(-1, 0)], dataMiddle[IDX(-1, 1)],
                                            w[y], dataMiddle[IDX(0, 0)], dataMiddle[IDX(0, 1)],
                                            w[y + 1], dataMiddle[IDX(1, 0)], dataMiddle[IDX(1, 1)]);
            }
            for (auto x = 1u; x < width - 1; x++) {
                dataMiddle[IDX(-1, 0)] = stencil(dataMiddle[IDX(-1, -1)], dataMiddle[IDX(-1, 0)], dataMiddle[IDX(-1, 1)],
                                                 dataMiddle[IDX(0, -1)], dataMiddle[IDX(0, 0)], dataMiddle[IDX(0, 1)],
                                                 dataMiddle[IDX(1, -1)], dataMiddle[IDX(1, 0)], dataMiddle[IDX(1, 1)])
            }
            {
                constexpr x = width - 1u;
                borderRight[height - 2] = stencil(dataMiddle[IDX(-1, -1)], dataMiddle[IDX(-1, 0)], e[y - 1],
                                                  dataMiddle[IDX(0, -1)], dataMiddle[IDX(0, 0)], e[y],
                                                  dataMiddle[IDX(1, -1)], dataMiddle[IDX(1, 0)], e[y + 1])
            }
        }


        // bottom row special case (we have to deal with sw, s and se here, but write to dataMiddle)
        {
            constexpr auto y = height - 1;
            { // sw
                constexpr auto x = 0u;
                borderLeft[height - 3] = stencil(w[height - 2], dataMiddle[IDX(-2, 0)], data[IDX(-2, 1)],
                                                 w[height - 1], dataBottom[0], dataBottom[1],
                                                 sw[0], s[0], s[1]);
            }
            // s
            for (auto x = 1u; x < width - 1; x++) {
                dataMiddle[IDX(-2, 0)] = stencil(dataMiddle[IDX(-2, -1)], data[IDX(-2, 0)], data[IDX(-2, 1)],
                                                 dataBottom[x - 1], dataBottom[x], dataBottom[x + 1],
                                                 s[x - 1], s[x], s[x + 1])
            }
            {  // se
                constexpr auto x = width - 1;
                borderRight[height - 3] = stencil(dataMiddle[IDX(-2, -1)], dataMiddle[IDX(-2, 0)], e[height - 2],
                                                  dataBottom[x - 1], dataBottom[x], e[height - 1],
                                                  ßs[x - 1], s[x], se[0])
            }

        }
        return true;
    }
};

template
class ScanDown<Float>;

template<typename T>
class ScanUp : public Vertex {

public:
    InOut <Vector<T, VectorLayout::ONE_PTR, 8>> dataMiddle; //(only the middle bit!)
    Output <Vector<T, VectorLayout::ONE_PTR, 8>> dataTop;
    Output <Vector<T, VectorLayout::ONE_PTR, 8>> dataBottom;
    Output <Vector<T, VectorLayout::ONE_PTR, 8>> dataLeft;
    Output <Vector<T, VectorLayout::ONE_PTR, 8>> dataRight;
    Input <Vector<T, VectorLayout::ONE_PTR, 8>> nw, ne, sw, se;
    Input <Vector<T, VectorLayout::ONE_PTR, 8>> n, s, w, e;
    Input <Vector<T, VectorLayout::ONE_PTR, 8>> borderLeft;
    Input <Vector<T, VectorLayout::ONE_PTR, 8>> borderRight;
    Input <Vector<T, VectorLayout::ONE_PTR, 8>> borderTop;
    Input <Vector<T, VectorLayout::ONE_PTR, 8>> borderBottom;
    unsigned width; // including borders
    unsigned height;

    bool compute() {

        // "bottom" row of partition
        { // halo to top is w, leftBorder, data, rightBorder, e
            // read my vals from w, leftBorder,data,rightBorder, e
            // read by bottom from sw, s, se
            constexpr auto y = height - 2;
            {
                constexpr x = 0u;
                data[]
            }
            for (auto x = 1u; x < width; x++) {

            }
            {
                constexpr x = width - 1;
            }
        }

        // second row
        // read from in borderTop , north neighbour is haloBottom, south is data[0], save to data[0]

        // first row
        // Was actually stored in borderBottom. It's halo is NW, N and NE to the top and to the south is borderTop in data
        {

        }


        return true;
    }
}

template
class ScanUp<float>;