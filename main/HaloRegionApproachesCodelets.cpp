#include <poplar/Vertex.hpp>
#include <cstddef>
#include <print.h>
#include <math.h>
#include <ipudef.h>


using namespace poplar;

template<typename T>
T stencil(const T nw, const T n, const T ne, const T w, const T m,
          const T e, const T sw,
          const T s, const T se) {
    return (nw + n + ne + w + m + e + sw + s + se) / 9;
}


template<typename T>
class Fill : public Vertex {

public:
    Output <Vector<T>> result;
    Input <T> val;

    bool compute() {
        for (auto i = 0; i < result.size(); i++) result[i] = *val;
        return true;
    }
};

template
class Fill<float>;

template<typename T>
class IncludedHalosApproach : public Vertex {

public:
    Input <VectorList<T, poplar::VectorListLayout::COMPACT_DELTAN, 4, false>> in;
    Output <VectorList<T, poplar::VectorListLayout::COMPACT_DELTAN, 4, false>> out;

    // Average the moore neighbourhood of the non-ghost part of the block
    bool compute() {
        // Only works if this is at least a 3x3 block, and in must be same size as out
        if (out.size() == in.size() && in.size() > 2 && in[0].size() > 2 && in[0].size() == out[0].size()) {
            for (auto y = 1u; y < in.size() - 1; y++) {
                for (auto x = 1u; x < in[y].size() - 1; x++) {
                    out[y][x] = stencil(in[y - 1][x - 1], in[y - 1][x], in[y - 1][x + 1],
                                        in[y][x - 1], in[y][x], in[y][x + 1],
                                        in[y + 1][x - 1], in[y + 1][x], in[y + 1][x + 1]);
                }
            }
            return true;
        }
        return false;
    }
};

template
class IncludedHalosApproach<float>;


template<typename T>
class ExtraHalosApproach : public Vertex {

public:
    Input <VectorList<T, poplar::VectorListLayout::COMPACT_DELTAN, 4, false>> in;
    Input <T> nw, ne, sw, se;
    Input <Vector<T>> n, s, w, e;
    Output <VectorList<T, poplar::VectorListLayout::COMPACT_DELTAN, 4, false>> out;

    // Average the moore neighbourhood of the non-ghost part of the block
    bool compute() {
        // Only works if this is at least a 3x3 block (excluding halos), and in must be same size as out
        if (out.size() == in.size() && in.size() > 2 && in[0].size() > 2 && in[0].size() == out[0].size()) {
            const auto nx = in[0].size();
            const auto ny = in.size();

            // top left
            out[0][0] = stencil(*nw, n[0], n[1], w[0], in[0][0], in[0][1], w[1], in[1][0], in[1][1]);

            // top
            for (auto x = 1u; x < in[0].size() - 1; x++) {
                out[0][x] =
                        (n[x - 1], n[x], n[x + 1], in[0][x - 1], in[0][x], in[0][x + 1], in[1][x - 1], in[1][x],
                                in[1][x + 1]);
            }

            // top right
            out[0][nx - 1] =
                    (n[nx - 2], n[nx - 1], *ne, in[0][nx - 2], in[0][nx - 1], e[0], in[1][nx - 2], in[1][nx - 1],
                            e[1]);


            // left col
//            for (y = 1; y < ny - 1; y++) {
//                out[y][0] = w[y - 1] + in[y - 1][0] + in[y - 1][1] +
//            }
            // middle block

            // right col

            // bottom left

            // bottom

            // bottom right
            return true;
        }
        return false;
    }
};

template
class ExtraHalosApproach<float>;


