//
// Created by Thorben Louw on 25/06/2020.
//

// Some util functions for structure grid applications (mapping tensors and distributing work optimally)

#ifndef LBM_GRAPHCORE_STRUCTUREGRIDUTILS_H
#define LBM_GRAPHCORE_STRUCTUREGRIDUTILS_H

#include <optional>
#include <cassert>
#include <memory>
#include <iostream>
#include <iomanip>
#include <utility>
#include <map>
#include <cmath>
#include <functional>


using namespace std;
namespace grids {

    /** Represents a range [inclusive..exclusive) */
    class Range {
    private:
        size_t t_from;
        size_t t_to;
    public:

        Range(size_t from, size_t to) : t_from(from), t_to(to) {
            assert(to > from);
        };

        size_t from() { return t_from; }

        size_t to() { return t_to; }
    };


    class Size2D {
    private:
        size_t t_rows;
        size_t t_cols;
    public:

        Size2D(size_t rows, size_t cols) : t_rows(rows), t_cols(cols) {
            assert(rows > 0);
            assert(cols > 0);
        }

        size_t rows() const { return t_rows; }

        size_t cols() const { return t_cols; }
    };

    class Slice2D {
    private:
        Range t_rows;
        Range t_cols;
        size_t t_width;
        size_t t_height;
    public:
        Slice2D(Range rows, Range cols) : t_rows(rows), t_cols(cols), t_width(cols.to() - cols.from()),
                                          t_height(rows.to() - rows.from()) {}

        Range rows() const { return t_rows; }

        Range cols() const { return t_cols; }

        size_t width() const { return t_width; }

        size_t height() const { return t_height; }
    };

    constexpr auto DefaultNumTilesPerIpu = 1216u;
    constexpr auto DefaultNumWorkersPerTile = 6u;
    constexpr auto DefaultMinRowsPerTile = 6u;
    constexpr auto DefaultMinColsPerTile = 6u;


    class MappingTarget {
        size_t t_ipu;
        size_t t_tile;
        size_t t_worker;
    public:


        explicit MappingTarget(size_t tile, size_t worker = 0, size_t ipu = 0) :
                t_ipu(ipu),
                t_tile(tile),
                t_worker(worker) {};

        size_t ipu() const { return t_ipu; }

        size_t tile() const { return t_tile; }

        size_t worker() const { return t_worker; }

    };


    struct MappingTargetComparator {
        bool operator()(const grids::MappingTarget &lhs, const grids::MappingTarget &rhs) const {
            auto l = lhs.ipu() * grids::DefaultNumTilesPerIpu * grids::DefaultNumWorkersPerTile +
                     lhs.tile() * grids::DefaultNumWorkersPerTile + lhs.worker();
            auto r = rhs.ipu() * grids::DefaultNumTilesPerIpu * grids::DefaultNumWorkersPerTile +
                     rhs.tile() * grids::DefaultNumWorkersPerTile + rhs.worker();
            return l < r;
        }
    };


    typedef std::map<MappingTarget, Slice2D, MappingTargetComparator> TileMappings;

    /**
     * A problem size so small we just use one tile
     */
    auto singleTileStrategy(Size2D size) -> TileMappings {
        TileMappings result;
        auto entry = Slice2D{{0, size.rows()},
                             {0, size.cols()}};
        auto key = MappingTarget{0};
        result.insert({key, entry});
        return result;
    };

    /**
    * A problem size small enough to just put the minimum size grid on as many tiles as needed
    */
    auto minSizeGridStrategy(Size2D size, size_t minRowsPerTile = DefaultMinRowsPerTile,
                             size_t minColsPerTile = DefaultMinColsPerTile) -> TileMappings {
        auto r = 0u;
        auto tile = 0u;
        auto tileMappings = TileMappings{};
        while (r < size.rows()) {
            auto r_end = min(r + minRowsPerTile, size.rows());
            auto c = 0u;
            while (c < size.cols()) {
                auto c_end = min(c + minColsPerTile, size.cols());
                tileMappings.insert({MappingTarget{tile}, {{r, r_end},
                                                           {c, c_end}}});
                tile++;
                c += minColsPerTile;
            }
            r += minRowsPerTile;
        }
        return tileMappings;
    };

    /**
   * The number of cols is less than the minimum but there are many rows, so we chunk vertically, respecting
     the min rows per tile given
   */
    auto longAndNarrowStrategy(Size2D size, size_t numTiles,
                               size_t minRowsPerTile = DefaultMinRowsPerTile) -> TileMappings {

        auto numTilesWhenUsingMinRowsConstraint = size.rows() / minRowsPerTile;
        auto numTilesToUse = min(numTiles, numTilesWhenUsingMinRowsConstraint);
        auto tileMappings = TileMappings{};

        auto numRowsPerTile = (size.rows() / numTilesToUse);
        auto numTilesWithExtra = size.rows() - (numTilesToUse * numRowsPerTile);
        auto r = 0ul;
        for (auto tile = 0ul; tile < numTilesToUse; tile++) {
            auto extra = tile < numTilesWithExtra;
            auto numRows = numRowsPerTile + extra;
            assert(r < size.rows());
            assert(r + numRows <= size.rows());

            tileMappings.insert({MappingTarget{tile}, {{r, r + numRows},
                                                       {0, size.cols()}}});
            r += numRows;
        }

        return tileMappings;
    };

    /**
   * The number of rows is less than the minimum, but there are many columns. We chunk it horizontally
   */
    auto shortAndWideStrategy(Size2D size, size_t numTiles,
                              size_t minColsPerTile = DefaultMinColsPerTile) -> TileMappings {

        auto numTilesWhenUsingMinColsConstraint = size.cols() / minColsPerTile;
        auto numTilesToUse = min(numTiles, numTilesWhenUsingMinColsConstraint);
        auto tileMappings = TileMappings{};

        auto c = 0ul;
        auto numColsPerTile = (size.cols() / numTilesToUse);
        auto numTilesWithExtra = size.cols() - (numTilesToUse * numColsPerTile);
        for (auto tile = 0ul; tile < numTilesToUse; tile++) {
            auto extra = tile < numTilesWithExtra;
            auto numCols = numColsPerTile + extra;
            assert(c < size.cols());
            assert(c + numCols <= size.cols());


            tileMappings.insert({MappingTarget{tile}, {{0, size.rows()},
                                                       {c, c + numCols}}});
            c += numCols;
        }

        return tileMappings;
    };

    /**
     * The general case grid decomposition for large problems on one ipu
     */
    auto generalGridStrategy(Size2D size, size_t numTiles, size_t minRowsPerTile = DefaultMinRowsPerTile,
                             size_t minColsPerTile = DefaultMinColsPerTile) -> TileMappings {
        double aspect_ratio = static_cast<double>(size.cols()) / static_cast<double> (size.rows());

        size_t tile_rows = min(numTiles, static_cast<size_t>(max(1., ceil(sqrt(numTiles / aspect_ratio)))));
        size_t tile_cols = min(numTiles, static_cast<size_t>(max(1., floor(numTiles / tile_rows))));

        auto nonwide_width = size.cols() / tile_cols;
        auto wide_width = nonwide_width + 1;
        auto num_wide_cols = size.cols() - tile_cols * nonwide_width;
        auto num_nonwide_cols = (tile_cols > num_wide_cols) ? tile_cols - num_wide_cols : 0;

        auto nontall_height = size.rows() / tile_rows;
        auto tall_height = nontall_height + 1;
        auto num_tall_rows = size.rows() - tile_rows * nontall_height;
        auto num_nontall_rows = (tile_rows > num_tall_rows) ? tile_rows - num_tall_rows : 0;

        auto tile = 0u;
        auto row_from = 0u;
        auto tileMapping = TileMappings{};
        for (size_t j = 0u; j < num_tall_rows; j++) {
            auto col_from = 0u;
            auto row_to = row_from + tall_height;

            auto col_to = col_from + wide_width;
            for (size_t i = 0u; i < num_wide_cols; i++) {
                assert(row_from < size.rows());
                assert(row_to <= size.rows());
                assert(col_from < size.cols());
                assert(col_to <= size.cols());
                tileMapping.insert({MappingTarget{tile}, {{row_from, row_to}, {col_from, col_to}}});
                col_from += wide_width;
                col_to = col_from + wide_width;
                tile++;
            }

            col_to = col_from + nonwide_width;
            for (size_t i = 0u; i < num_nonwide_cols; i++) {
                assert(row_from < size.rows());
                assert(row_to <= size.rows());
                assert(col_from < size.cols());
                assert(col_to <= size.cols());
                tileMapping.insert({MappingTarget{tile}, {{row_from, row_to}, {col_from, col_to}}});
                col_from += nonwide_width;
                col_to = col_from + nonwide_width;
                tile++;
            }
            row_from += tall_height;
        }

        for (size_t j = 0u; j < num_nontall_rows; j++) {
            auto col_from = 0u;
            auto row_to = row_from + nontall_height;
            auto col_to = col_from + wide_width;
            for (size_t i = 0u; i < num_wide_cols; i++) {
                assert(row_from < size.rows());
                assert(row_to <= size.rows());
                assert(col_from < size.cols());
                assert(col_to <= size.cols());
                tileMapping.insert({MappingTarget{tile}, {{row_from, row_to}, {col_from, col_to}}});
                col_from += wide_width;
                col_to = col_from + wide_width;
                tile++;
            }

            col_to = col_from + nonwide_width;
            for (size_t i = 0u; i < num_nonwide_cols; i++) {
                assert(row_from < size.rows());
                assert(row_to <= size.rows());
                assert(col_from < size.cols());
                assert(col_to <= size.cols());
                tileMapping.insert({MappingTarget{tile}, {{row_from, row_to}, {col_from, col_to}}});
                col_from += nonwide_width;
                col_to = col_from + nonwide_width;
                tile += 1;
            }
            row_from += nontall_height;

        }

        return tileMapping;
    }


    /**
     * Split a tile's workload into roughly equal chunks for the 6 workers. We try to assign chunks of rows,
     * but if there are more than 6x cols than rows we switch to a longAndTall strategy and chunk into cols
     */
    auto workerMappingForTile(MappingTarget target, Slice2D slice,
                              size_t numWorkersPerTile = DefaultNumWorkersPerTile) -> TileMappings {
        const auto tile = target.tile();
        const auto ipu = target.ipu();
        TileMappings workerMappings = {};

        const auto useShortAndWideStrategy = slice.width() >= slice.height() * numWorkersPerTile;

        if (useShortAndWideStrategy) {
            auto numWorkersToUse = min(slice.width(), numWorkersPerTile);

            auto numColsPerWorker = (slice.width() / numWorkersToUse);
            auto numWorkersWithExtra = slice.width() - (numWorkersToUse * numColsPerWorker);
            auto c = slice.cols().from();
            for (auto worker = 0ul; worker < numWorkersToUse; worker++) {
                auto extra = worker < numWorkersWithExtra;
                auto numCols = numColsPerWorker + extra;
                assert(c < slice.cols().to());
                assert(c + numCols <= slice.cols().to());

                workerMappings.insert({MappingTarget{tile, worker, ipu}, {
                        {slice.rows().from(), slice.rows().to()}, {c, c + numCols}
                }});
                c += numCols;
            }
        } else {
            auto numWorkersToUse = min(slice.height(), numWorkersPerTile);

            auto numRowsPerWorker = (slice.height() / numWorkersToUse);
            auto numWorkersWithExtra = slice.height() - (numWorkersToUse * numRowsPerWorker);
            auto r = slice.rows().from();
            for (auto worker = 0ul; worker < numWorkersToUse; worker++) {
                auto extra = worker < numWorkersWithExtra;
                auto numRows = numRowsPerWorker + extra;
                assert(r < slice.rows().to());
                assert(r + numRows <= slice.rows().to());

                workerMappings.insert({MappingTarget{tile, worker, ipu}, {{r, r + numRows},
                                                             {slice.cols().from(), slice.cols().to()}}});
                r += numRows;
            }
        }
        return workerMappings;
    }


    /**
     * As an intermediate step in mapping down to worker split, determine the split down to tile level.
     * All MappingTargets will have worker=0. Use toWorkerMappings to further refine down to worker split
     */
    auto partitionGridToTilesForSingleIpu(Size2D size,
                                          size_t numTiles = DefaultNumTilesPerIpu,
                                          size_t minRowsPerTile = DefaultMinRowsPerTile,
                                          size_t minColsPerTile = DefaultMinColsPerTile) -> TileMappings {
        if (size.cols() * size.rows() < minColsPerTile * minRowsPerTile) {
            // This is unlikely for a real case! Not even going to try and optimise for it
            return singleTileStrategy(size);
        } else if (size.cols() < minColsPerTile) {
            // We have something that's narrow but long, so chop it up by rows
            return longAndNarrowStrategy(size, numTiles, minRowsPerTile);
        } else if (size.rows() < minRowsPerTile) {
            // We have something that's wide but not long, so chop it up by cols
            return shortAndWideStrategy(size, numTiles, minColsPerTile);
        } else if (size.cols() * size.rows() < numTiles * minColsPerTile * minRowsPerTile) {
            // We'll use tiles of 64x6
            return minSizeGridStrategy(size, minRowsPerTile, minColsPerTile);
        } else {
            // We'll try and use the best grid overlay we can
            return generalGridStrategy(size, numTiles, minRowsPerTile, minColsPerTile);
        }
    }

    /**
     * Further splits a tile mapping that is the result of @refitem  partitionGridToTileForSingleIpu futher into worker mappings
     */
    auto toWorkerMappings(const TileMappings &tileMappings,
                          size_t numWorkersPerTile = DefaultNumWorkersPerTile) -> TileMappings {
        TileMappings result = {};
        for (const auto&[target, tileSlice]: tileMappings) {
            auto newMappings = workerMappingForTile(target, tileSlice, numWorkersPerTile);
            for (auto &[newTarget, newTileSlice]: newMappings) {
                result.insert({newTarget, newTileSlice});
            }
        }
        return result;
    }

    class Halos {
    public:
        const std::optional<Slice2D> top, bottom, left, right, topLeft, topRight, bottomLeft, bottomRight;

        Halos() = delete;

        Halos(std::optional<Slice2D> top,
              std::optional<Slice2D> bottom,
              std::optional<Slice2D> left,
              std::optional<Slice2D> right,
              std::optional<Slice2D> topLeft,
              std::optional<Slice2D> topRight,
              std::optional<Slice2D> bottomLeft,
              std::optional<Slice2D> bottomRight) :
                top(std::move(top)), bottom(std::move(bottom)), left(std::move(left)), right(std::move(right)),
                topLeft(std::move(topLeft)), topRight(std::move(topRight)),
                bottomLeft(std::move(bottomLeft)), bottomRight(std::move(bottomRight)) {
        }

        static auto forSlice(Slice2D slice, Size2D matrixSize, bool wraparound = true) -> Halos {
            // Some shorthand sugar
            const auto x = slice.cols().from();
            const auto y = slice.rows().from();
            const auto w = slice.width();
            const auto h = slice.height();
            const auto nx = matrixSize.cols();
            const auto ny = matrixSize.rows();

            std::optional<size_t> t, l, r, b;
            if (wraparound) {
                t = (ny + y - 1) % ny;
                l = (nx + x - 1) % nx;
                r = (nx + x + w) % nx;
                b = (ny + y + h) % ny;
            } else {
                t = (y > 0) ? std::optional<size_t>{y - 1} : std::nullopt;
                l = (x > 0) ? std::optional<size_t>{x - 1} : std::nullopt;
                r = (x + w < nx) ? std::optional<size_t>{x + w} : std::nullopt;
                b = (y + h < ny) ? std::optional<size_t>{y + h} : std::nullopt;
            }

            auto topLeft = (l.has_value() && t.has_value())
                           ? std::optional<Slice2D>{
                            {{*t, *t + 1},
                                    {*l, *l + 1}}}
                           : std::nullopt;
            auto top = (t.has_value())
                       ? std::optional<Slice2D>{
                            {{*t, *t + 1},
                                    {*l, *l + 1}}}
                       : std::nullopt;

            auto topRight = (t.has_value() && r.has_value())
                            ? std::optional<Slice2D>{
                            {
                                    {*t, *t + 1},
                                    {*r, *r + 1}
                            }}
                            : std::nullopt;

            auto left = (l.has_value()) ? std::optional<Slice2D>{
                    {{y, y + h},
                            {*l, *l + 1}}} : std::nullopt;
            auto right = r.has_value()
                         ? std::optional<Slice2D>{{{y, y + h},
                                                          {*r, *r + 1}}}
                         : std::nullopt;
            auto bottomLeft = (l.has_value() && b.has_value())
                              ? std::optional<Slice2D>{{{*b, *b + 1},
                                                               {*l, *l + 1}}}
                              : std::nullopt;
            auto bottom = (b.has_value())
                          ? std::optional<Slice2D>{
                            {{*b, *b + 1},
                                    {x, x + w}}}
                          : std::nullopt;
            auto bottomRight = (b.has_value() && r.has_value())
                               ? std::optional<Slice2D>{
                            {
                                    {*b, *b + 1},
                                    {*r, *r + 1}
                            }} : std::nullopt;
            return Halos(top, bottom, left, right, topLeft, topRight, bottomLeft, bottomRight);
        }

    };


}



//
//average tile dimension:
//say we have a rows * cols matrix
//
//for 128x128 we have 16384 things to process
//we divide them among 1216 tiles as evenly as possible
//everything gets 13-14 things to process
//try and do the same number of rows but just more cols for the extras
//
//(we can just plan by number of workers also - 7296. In this case every worker has 2-3 cells to work on)
//every worker just works on a chunk of a row. never use a grid piece?



//
//then we
//get a
//rows/1216 * cols/1216
//matrix on
//every tile

//void print_matrix(const std::unique_ptr<float[]> &matrix, const unsigned numRows, const unsigned numCols,
//                  const unsigned numRowsAndColsToPrint = 10) {
//    for (unsigned i = 0; i < std::min(numRows, numRowsAndColsToPrint); i++) {
//        for (unsigned j = 0; j < std::min(numCols, numRowsAndColsToPrint); j++) {
//            std::cout << std::fixed << std::setw(5) << std::setprecision(5) << matrix[i * numCols + j] << " ";
//        }
//        std::cout << std::endl;
//    }
//}
//
//
//unsigned numRowsForTile(unsigned totalNumRows, unsigned tileNum, unsigned totalNumTiles) {
//    auto extra = tileNum < totalNumRows % totalNumTiles;
//    return unsigned(totalNumRows / totalNumTiles) + extra;
//}


///**
// * these are UNpadded col sizes, but returns PADDED indices. num rows includes padding? Just how many
// * rows this tile will be processing
// */
//const StartEnds determineColumnChunkSplits(const unsigned rows, const unsigned cols) {
//
//    // if we can perfectly divide over the 6 threads, it's a happy day
//    if ((rows * cols) % 6 == 0) {
//        return {{0, cols + 1}};
//    }
//
//    // We try and create rows * x tasks so that (rows * x) % 6 == 0, or as close as, while keeping
//    // the number of vertexes sane . We don't want more than 60? vertexes in total (arbitrary!)
//
//    unsigned divisor = 6u;
//
//    // std::cout << "Using " << divisor << " chunks" << std::endl;
//    auto splits = StartEnds{};
//    unsigned currentStartPos = 1;
//    unsigned PADDING_START = -1;
//    unsigned PADDING_END = 1;
//    for (auto i = 0u; i < divisor; i++) {
//        auto extra = i < cols % divisor;
//        auto numCols = (cols / divisor) + extra;
//        auto start = PADDING_START + currentStartPos;
//        auto end = start + numCols + PADDING_END;
//        // std::cout << i<< " is chunk of " << numCols << " from " << start << " to " << end << std::endl;
//        splits.push_back({start, end});
//        currentStartPos += numCols;
//    }
//
//    return splits;
//}
//
///**
// * Maps the two image tensors' rows onto the tiles that will be processing them, to reduce
// * exchanges
// */
//void applyTileMappingsForTensors(Tensor &t1, Tensor &t2, Graph &graph, const TileMappings &tileMappings) {
//    assert(t1.dim(0) == t2.dim(0));
//    assert(t1.dim(1) == t2.dim(1));
//
//    for (auto const &mapping : tileMappings) {
//        const auto tile = mapping.first;
//        const auto startRow = mapping.second.first;
//        const auto endRow = mapping.second.second; // note slice's second param is non-inclusive i.e. [start, end)
//        graph.setTileMapping(t1.slice(startRow, endRow + 1, 0), tile);
//        graph.setTileMapping(t2.slice(startRow, endRow + 1, 0), tile);
//    }
//}
//
///**
// * Maps the two image tensors' rows onto the tiles that will be processing them, to reduce
// * exchanges
// * Returns a map of tileNum -> (startRow, endRow) [both inclusive]
// * numRows is unpadded size
// */
//TileMappings getTensorRowMappingToTiles(unsigned numRows, unsigned numTilesToUse) {
//    unsigned currentRow = 0;
//
//    TileMappings result;
//    for (unsigned tile = 0; tile < numTilesToUse; tile++) {
//        auto numRowsForThisTile = numRowsForTile(numRows, tile, numTilesToUse);
//        auto startRow = (tile == 0u) ? 0 : currentRow + 1; // The first tile gets the padded 0 row as well
//        auto endRow = (tile == numTilesToUse - 1) ? numRows + 1 : currentRow +
//                                                                  numRowsForThisTile; // the last tile gets the padded bottom row as well
//        auto val = std::pair(startRow, endRow);
//        result.insert(std::pair(tile, val));
//        currentRow += numRowsForThisTile;
//    }
//
//    return result;
//}
//
///**
// * For debug, prints the tile mapping for the stencil tensor
// */
//void printTileMappings(const TileMappings &tileMappings) {
//    for (const auto &tileMappings : tileMappings) {
//        const auto tile = tileMappings.first;
//        const auto fromRow = tileMappings.second.first;
//        const auto toRow = tileMappings.second.second;
//        std::cout << "[" << fromRow << ":" << toRow << "] -> " << tile << std::endl;
//    }
//}

#endif //LBM_GRAPHCORE_STRUCTUREGRIDUTILS_H
