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
#include <fstream>


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

        [[nodiscard]]  const size_t from() const { return t_from; }

        [[nodiscard]] const size_t to() const { return t_to; }
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

        [[nodiscard]] size_t rows() const { return t_rows; }

        [[nodiscard]] size_t cols() const { return t_cols; }
    };

    class Slice2D {
    private:
        Range t_rows;
        Range t_cols;
        size_t t_width;
        size_t t_height;
    public:
        Slice2D(const Range rows, const Range cols) : t_rows(rows), t_cols(cols), t_width(cols.to() - cols.from()),
                                                      t_height(rows.to() - rows.from()) {}

        [[nodiscard]] Range rows() const { return t_rows; }

        [[nodiscard]] Range cols() const { return t_cols; }

        [[nodiscard]] size_t width() const { return t_width; }

        [[nodiscard]] size_t height() const { return t_height; }

        [[nodiscard]] Size2D size() const { return Size2D{t_width, t_height}; }
    };

    constexpr auto DefaultNumTilesPerIpu = 1216u;
    constexpr auto DefaultNumWorkersPerTile = 6u;
    constexpr auto DefaultMinRowsPerTile = 6u;
    constexpr auto DefaultMinColsPerTile = 6u;


    class PartitioningTarget {
        size_t t_ipu;
        size_t t_tile;
        size_t t_worker;
    public:


        explicit PartitioningTarget(size_t ipu = 0, size_t tile = 0, size_t worker = 0) :
                t_ipu(ipu),
                t_tile(tile),
                t_worker(worker) {};

        [[nodiscard]] size_t ipu() const { return t_ipu; }

        [[nodiscard]] size_t tile() const { return t_tile; }

        [[nodiscard]] size_t worker() const { return t_worker; }

        [[nodiscard]] size_t virtualTile(size_t numTilesPerIpu = DefaultNumTilesPerIpu) const {
            return t_ipu * numTilesPerIpu + t_tile;
        }


    };


    struct PartitionTargetComparator {
        bool operator()(const grids::PartitioningTarget &lhs, const grids::PartitioningTarget &rhs) const {
            auto l = lhs.ipu() * grids::DefaultNumTilesPerIpu * grids::DefaultNumWorkersPerTile +
                     lhs.tile() * grids::DefaultNumWorkersPerTile + lhs.worker();
            auto r = rhs.ipu() * grids::DefaultNumTilesPerIpu * grids::DefaultNumWorkersPerTile +
                     rhs.tile() * grids::DefaultNumWorkersPerTile + rhs.worker();
            return l < r;
        }
    };


    typedef std::map<PartitioningTarget, Slice2D, PartitionTargetComparator> GridPartitioning;

    auto serializeToJson(const GridPartitioning &partitioning, const std::string &filename) {
        ofstream file;
        file.open(filename);
        file << R"EOS({"GridPartitioning" : [)EOS" << std::endl;
        bool comma = false;
        for (const auto&[target, slice]: partitioning) {
            if (comma) {
                file << "," << std::endl;
            }
            file << "  {" << std::endl;
            file << R"'(    "ipu":)'" << target.ipu() << "," << std::endl;
            file << R"'(    "tile":)'" << target.tile() << "," << std::endl;
            file << R"'(    "worker":)'" << target.worker() << "," << std::endl;
            file << R"'(    "slice": {)'" << std::endl;
            file << R"'(       "rows" : { "from" : )'" << slice.rows().from() << "," <<
                 R"'("to" : )'" << slice.rows().to() << "}," << std::endl;
            file << R"'(       "cols" : { "from" : )'" << slice.cols().from() << "," <<
                 R"'("to" : )'" << slice.cols().to() << "}" << std::endl;
            file << "  }";
            comma = true;
        }
        file << std::endl << "]}" << std::endl;
        file.close();
    }

    /**
     * A problem size so small we just use one tile
     */
    auto singleTileStrategy(const PartitioningTarget target, const Slice2D slice) -> GridPartitioning {
        GridPartitioning result;
        const auto key = PartitioningTarget{target.ipu(), 0};
        result.insert({key, slice});
        return result;
    }


    auto singleIpuStrategy(Size2D size) -> GridPartitioning {
        GridPartitioning result;
        auto entry = Slice2D{{0, size.rows()},
                             {0, size.cols()}};
        auto key = PartitioningTarget{0};
        result.insert({key, entry});
        return result;
    }

    /**
    * A problem size small enough to just put the minimum size grid on as many tiles as needed
    */
    auto minSizeTileGridStrategy(const PartitioningTarget target, const Slice2D slice,
                                 const size_t minRowsPerTile = DefaultMinRowsPerTile,
                                 const size_t minColsPerTile = DefaultMinColsPerTile) -> GridPartitioning {
        auto r = 0u;
        auto tile = 0u;
        auto partitioning = GridPartitioning{};
        while (r < slice.height()) {
            auto r_end = min(r + minRowsPerTile, slice.height());
            auto c = 0u;
            while (c < slice.width()) {
                auto c_end = min(c + minColsPerTile, slice.width());
                partitioning.insert({PartitioningTarget{target.ipu(), tile},
                                     {{r + slice.rows().from(), r_end + slice.rows().from()},
                                      {c + slice.cols().from(), c_end + slice.cols().from()}}});
                tile++;
                c += minColsPerTile;
            }
            r += minRowsPerTile;
        }
        return partitioning;
    }

    /**
   * The number of cols is less than the minimum but there are many rows, so we chunk vertically, respecting
     the min rows per tile given
   */
    auto longAndNarrowTileStrategy(const PartitioningTarget target, const Slice2D slice, const size_t numTiles,
                                   const size_t minRowsPerTile = DefaultMinRowsPerTile) -> GridPartitioning {

        auto numTilesWhenUsingMinRowsConstraint = slice.height() / minRowsPerTile;
        auto numTilesToUse = min(numTiles, numTilesWhenUsingMinRowsConstraint);
        auto partitioning = GridPartitioning{};

        auto numRowsPerTile = (slice.height() / numTilesToUse);
        auto numTilesWithExtra = slice.height() - (numTilesToUse * numRowsPerTile);
        auto r = 0ul;
        for (auto tile = 0ul; tile < numTilesToUse; tile++) {
            auto extra = tile < numTilesWithExtra;
            auto numRows = numRowsPerTile + extra;
            assert(r < slice.height());
            assert(r + numRows <= slice.height());

            partitioning.insert(
                    {PartitioningTarget{target.ipu(), tile},
                     {{slice.rows().from() + r, slice.rows().from() + r + numRows},
                      {slice.cols().from(), slice.cols().to()}}});
            r += numRows;
        }

        return partitioning;
    }

    /**
 * The number of cols is less than the minimum but there are many rows, so we chunk vertically, respecting
   the min rows per tile given
 */
    auto longAndNarrowIpuStrategy(Size2D size, size_t numIpus,
                                  size_t maxCellsPerIpu = DefaultMinRowsPerTile) -> optional<GridPartitioning> {

        auto tileMappings = GridPartitioning{};

        auto numRowsPerTile = (size.rows() / numIpus);
        auto numIpusWithExtra = size.rows() - (numIpus * numRowsPerTile);
        auto r = 0ul;
        for (auto ipu = 0ul; ipu < numIpus; ipu++) {
            auto extra = ipu < numIpusWithExtra;
            auto numRows = numRowsPerTile + extra;
            assert(r < size.rows());
            assert(r + numRows <= size.rows());

            if (numRows * size.cols() > maxCellsPerIpu)
                return nullopt;// This chunk is too big! This strategy won't work

            tileMappings.insert({PartitioningTarget{ipu}, {{r, r + numRows},
                                                           {0, size.cols()}}});
            r += numRows;
        }

        return {tileMappings};
    }

    /**
   * The number of rows is less than the minimum, but there are many columns. We chunk it horizontally
   */
    auto shortAndWideTileStrategy(const PartitioningTarget target, const Slice2D slice, const size_t numTiles,
                                  const size_t minColsPerTile = DefaultMinColsPerTile) -> GridPartitioning {

        auto numTilesWhenUsingMinColsConstraint = slice.width() / minColsPerTile;
        auto numTilesToUse = min(numTiles, numTilesWhenUsingMinColsConstraint);
        auto tileMappings = GridPartitioning{};

        auto c = 0ul;
        auto numColsPerTile = (slice.width() / numTilesToUse);
        auto numTilesWithExtra = slice.width() - (numTilesToUse * numColsPerTile);
        for (auto tile = 0ul; tile < numTilesToUse; tile++) {
            auto extra = tile < numTilesWithExtra;
            auto numCols = numColsPerTile + extra;
            assert(c < slice.width());
            assert(c + numCols <= slice.width());


            tileMappings.insert(
                    {PartitioningTarget{target.ipu(), tile}, {{slice.rows().from(), slice.rows().to()},
                                                              {slice.cols().from() + c,
                                                               slice.cols().from() + c + numCols}}});
            c += numCols;
        }

        return tileMappings;
    };

    /**
 * We found that dividing work by cols is the best balance
 */
    auto shortAndWideIpuStrategy(const Size2D size, const size_t numIpus,
                                 const size_t maxCellsPerIpu) -> optional<GridPartitioning> {


        auto tileMappings = GridPartitioning{};

        auto c = 0ul;
        auto numColsPerIpu = (size.cols() / numIpus);
        auto numIpusWithExtra = size.cols() - (numIpus * numColsPerIpu);
        for (auto ipu = 0ul; ipu < numIpus; ipu++) {
            auto extra = ipu < numIpusWithExtra;
            auto numCols = numColsPerIpu + extra;
            assert(c < size.cols());
            assert(c + numCols <= size.cols());

            if (numCols * size.rows() > maxCellsPerIpu)
                return nullopt; // This chunk is too big! This strategy won't work

            tileMappings.insert({PartitioningTarget{ipu}, {{0, size.rows()},
                                                           {c, c + numCols}}});
            c += numCols;
        }

        return {tileMappings};
    };

    /**
     * The general case grid decomposition for large problems on one ipu
     */
    auto generalTileGridStrategy(const PartitioningTarget target,
                                 const Slice2D slice,
                                 const size_t numTiles,
                                 const size_t minRowsPerTile = DefaultMinRowsPerTile,
                                 const size_t minColsPerTile = DefaultMinColsPerTile) -> GridPartitioning {


        double aspect_ratio = static_cast<double>(max(minColsPerTile, slice.width())) /
                              static_cast<double> (max(minRowsPerTile, slice.height()));


        size_t tile_rows = min(numTiles,
                               static_cast<size_t>(max((double) minRowsPerTile, ceil(sqrt(numTiles / aspect_ratio)))));
        size_t tile_cols = min(numTiles,
                               static_cast<size_t>(max((double) minColsPerTile, floor(numTiles / tile_rows))));

        auto nonwide_width = slice.width() / tile_cols;
        auto wide_width = nonwide_width + 1;
        auto num_wide_cols = slice.width() - tile_cols * nonwide_width;
        auto num_nonwide_cols = (tile_cols > num_wide_cols) ? tile_cols - num_wide_cols : 0;

        auto nontall_height = slice.height() / tile_rows;
        auto tall_height = nontall_height + 1;
        auto num_tall_rows = slice.height() - tile_rows * nontall_height;
        auto num_nontall_rows = (tile_rows > num_tall_rows) ? tile_rows - num_tall_rows : 0;

        auto tile = 0u;
        auto row_from = 0u;
        auto tileMapping = GridPartitioning{};
        for (size_t j = 0u; j < num_tall_rows; j++) {
            auto col_from = 0u;
            auto row_to = row_from + tall_height;

            auto col_to = col_from + wide_width;
            for (size_t i = 0u; i < num_wide_cols; i++) {
                assert(row_from < slice.height());
                assert(row_to <= slice.height());
                assert(col_from < slice.width());
                assert(col_to <= slice.width());
                tileMapping.insert({PartitioningTarget{target.ipu(), tile},
                                    {{slice.rows().from() + row_from, slice.rows().from() + row_to},
                                     {slice.cols().from() + col_from, slice.cols().from() + col_to}}});
                col_from += wide_width;
                col_to = col_from + wide_width;
                tile++;
            }

            col_to = col_from + nonwide_width;
            for (size_t i = 0u; i < num_nonwide_cols; i++) {
                assert(row_from < slice.height());
                assert(row_to <= slice.height());
                assert(col_from < slice.width());
                assert(col_to <= slice.width());
                tileMapping.insert({PartitioningTarget{target.ipu(), tile},
                                    {{slice.rows().from() + row_from, slice.rows().from() + row_to},
                                     {slice.cols().from() + col_from, slice.cols().from() + col_to}}});
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
                assert(row_from < slice.height());
                assert(row_to <= slice.height());
                assert(col_from < slice.width());
                assert(col_to <= slice.width());
                tileMapping.insert({PartitioningTarget{target.ipu(), tile},
                                    {{slice.rows().from() + row_from, slice.rows().from() + row_to},
                                     {slice.cols().from() + col_from, slice.cols().from() + col_to}}});
                col_from += wide_width;
                col_to = col_from + wide_width;
                tile++;
            }

            col_to = col_from + nonwide_width;
            for (size_t i = 0u; i < num_nonwide_cols; i++) {
                assert(row_from < slice.height());
                assert(row_to <= slice.height());
                assert(col_from < slice.width());
                assert(col_to <= slice.width());
                tileMapping.insert({PartitioningTarget{target.ipu(), tile},
                                    {{slice.rows().from() + row_from, slice.rows().from() + row_to},
                                     {slice.cols().from() + col_from, slice.cols().from() + col_to}}});
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
    auto toWorkerPartitions(const PartitioningTarget target, const Slice2D slice,
                            const size_t numWorkersPerTile = DefaultNumWorkersPerTile) -> GridPartitioning {
        const auto tile = target.tile();
        GridPartitioning workerMappings = {};

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

                workerMappings.insert({PartitioningTarget{target.ipu(), tile, worker}, {
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

                workerMappings.insert({PartitioningTarget{target.ipu(), tile, worker}, {{r, r + numRows},
                                                                                        {slice.cols().from(),
                                                                                         slice.cols().to()}}});
                r += numRows;
            }
        }
        return workerMappings;
    }


    /**
     * As an intermediate step in mapping down to worker split, determine the split down to tile level.
     * All MappingTargets will have worker=0. Use toWorkerMappings to further refine down to worker split
     */
    auto partitionForIpus(Size2D size,
                          size_t numIpus,
                          size_t maxCellsPerIpu) -> optional<GridPartitioning> {
        // Lost cause! Too much data
        if (size.rows() * size.cols() > maxCellsPerIpu * numIpus) return nullopt;

        // It all fits on one IPU - best we can do is keep it on one to minimise cost of exchange
        if (size.cols() * size.rows() <= maxCellsPerIpu)
            return {singleIpuStrategy(size)};

        // How will work be more evenly split? By rows or by cols? We want most even
        float rowImbalance = (float) (size.rows() % numIpus) / (float) size.rows();
        float colImbalance = (float) (size.cols() % numIpus) / (float) size.cols();

        if (rowImbalance < colImbalance) {
            if (auto result = longAndNarrowIpuStrategy(size, numIpus, maxCellsPerIpu); result.has_value()) {
                return result;
            } else {
                // Couldn't partition. Try cols?
                return shortAndWideIpuStrategy(size, numIpus, maxCellsPerIpu);
            }
        } else {
            if (auto result = shortAndWideIpuStrategy(size, numIpus, maxCellsPerIpu); result.has_value()) {
                return result;
            } else {
                // Couldn't partition. Try rows?
                return longAndNarrowIpuStrategy(size, numIpus, maxCellsPerIpu);
            }
        }
    }


    /**
     * As an intermediate step in mapping down to worker split, determine the split down to tile level.
     * All MappingTargets will have worker=0. Use toWorkerMappings to further refine down to worker split
     */
    auto toTilePartitionsForSingleIpu(const PartitioningTarget target,
                                      const Slice2D slice,
                                      const size_t numTiles = DefaultNumTilesPerIpu,
                                      const size_t minRowsPerTile = DefaultMinRowsPerTile,
                                      const size_t minColsPerTile = DefaultMinColsPerTile) -> GridPartitioning {
        if (slice.width() * slice.height() < minColsPerTile * minRowsPerTile) {
            // This is unlikely for a real case! Not even going to try and optimise for it
            return singleTileStrategy(target, slice);
        } else if (slice.width() < minColsPerTile) {
            // We have something that's narrow but long, so chop it up by rows
            return longAndNarrowTileStrategy(target, slice, numTiles, minRowsPerTile);
        } else if (slice.height() < minRowsPerTile) {
            // We have something that's wide but not long, so chop it up by cols
            return shortAndWideTileStrategy(target, slice, numTiles, minColsPerTile);
        } else if (slice.width() * slice.height() < numTiles * minColsPerTile * minRowsPerTile) {
            // We'll use tiles of 64x6
            return minSizeTileGridStrategy(target, slice, minRowsPerTile, minColsPerTile);
        } else {
            // We'll try and use the best grid overlay we can
            return generalTileGridStrategy(target, slice, numTiles, minRowsPerTile, minColsPerTile);
        }
    }

    /**
     * Further splits a tile mapping that is the result of @refitem  partitionGridToTileForSingleIpu futher into worker mappings
     */
    auto toWorkerPartitions(const GridPartitioning &tileMappings,
                            size_t numWorkersPerTile = DefaultNumWorkersPerTile) -> GridPartitioning {
        GridPartitioning result = {};
        for (const auto&[target, tileSlice]: tileMappings) {
            auto newMappings = toWorkerPartitions(target, tileSlice, numWorkersPerTile);
            for (auto &[newTarget, newTileSlice]: newMappings) {
                result.insert({newTarget, newTileSlice});
            }
        }
        return result;
    }


    auto toTilePartitions(const GridPartitioning &ipuMappings,
                          const size_t numTiles = DefaultNumTilesPerIpu,
                          const size_t minRowsPerTile = DefaultMinRowsPerTile,
                          const size_t minColsPerTile = DefaultMinColsPerTile) {
        assert(ipuMappings.size() > 0);
        GridPartitioning result = {};
        for (const auto&[target, ipuSlice]: ipuMappings) {
            auto newMappings = toTilePartitionsForSingleIpu(target, ipuSlice, numTiles,
                                                            minRowsPerTile, minColsPerTile);
            for (const auto &[newTarget, newTileSlice]: newMappings) {
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


        // Top left is (0,0) as in Gaussian Blur
        static auto forSliceTopIs0NoWrap(Slice2D slice, Size2D matrixSize) -> Halos {
            // Some shorthand sugar
            const auto x = slice.cols().from();
            const auto y = slice.rows().from();
            const auto w = slice.width();
            const auto h = slice.height();
            const auto nx = matrixSize.cols();
            const auto ny = matrixSize.rows();

            int t, l;
            unsigned int r, b;
            t = (int) y - 1;
            l = (int) x - 1;
            r = x + w;
            b = y + h;

            auto topLeft = (t > 0 && l > 0)
                           ? std::optional<Slice2D>{
                            {{(unsigned) t, (unsigned) t + 1},
                                    {(unsigned) l, (unsigned) l + 1}}}
                           : std::nullopt;
            auto top = (t > 0)
                       ? std::optional<Slice2D>{
                            {{(unsigned) t, (unsigned) t + 1},
                                    {x, x + w}}}
                       : std::nullopt;

            auto topRight = (t > 0 && r < nx - 1)
                            ? std::optional<Slice2D>{
                            {
                                    {(unsigned) t, (unsigned) t + 1},
                                    {r, r + 1}
                            }}
                            : std::nullopt;

            auto left = (l > 0) ? std::optional<Slice2D>{
                    {{y, y + h},
                            {(unsigned) l, (unsigned) l + 1}}} : std::nullopt;
            auto right = r < nx - 1
                         ? std::optional<Slice2D>{{{y, y + h},
                                                          {r, r + 1}}}
                         : std::nullopt;
            auto bottomLeft = (l > 0 && b < ny - 1)
                              ? std::optional<Slice2D>{{{b, b + 1},
                                                               {(unsigned) l, (unsigned) l + 1}}}
                              : std::nullopt;
            auto bottom = (b < ny - 1)
                          ? std::optional<Slice2D>{
                            {{b, b + 1},
                                    {x, x + w}}}
                          : std::nullopt;
            auto bottomRight = (b < ny - 1) && (r < nx - 1)
                               ? std::optional<Slice2D>{
                            {
                                    {b, b + 1},
                                    {r, r + 1}
                            }} : std::nullopt;
            return Halos(top, bottom, left, right, topLeft, topRight, bottomLeft, bottomRight);

        }


        // IMPORTANT! (0,0) is considered bottom left (as in LBM )
        static auto
        forSliceBottomIs0Wraparound(Slice2D slice, Size2D matrixSize) -> Halos {
            // Some shorthand sugar
            const auto x = slice.cols().from();
            const auto y = slice.rows().from();
            const auto w = slice.width();
            const auto h = slice.height();
            const auto nx = matrixSize.cols();
            const auto ny = matrixSize.rows();

            std::optional<size_t> t, l, r, b;
            t = (ny + y + 1) % ny;
            l = (nx + x - 1) % nx;
            r = (nx + x + w) % nx;
            b = (ny + y - h) % ny;

            auto topLeft = std::optional<Slice2D>{
                    {{*t, *t + 1},
                            {*l, *l + 1}}};
            auto top = std::optional<Slice2D>{
                    {{*t, *t + 1},
                            {x, x + w}}};

            auto topRight = std::optional<Slice2D>{
                    {
                            {*t, *t + 1},
                            {*r, *r + 1}
                    }};

            auto left = std::optional<Slice2D>{
                    {{y, y + h},
                            {*l, *l + 1}}};
            auto right = std::optional<Slice2D>{{{y, y + h},
                                                        {*r, *r + 1}}};
            auto bottomLeft = std::optional<Slice2D>{{{*b, *b + 1},
                                                             {*l, *l + 1}}};
            auto bottom = std::optional<Slice2D>{
                    {{*b, *b + 1},
                            {x, x + w}}};
            auto bottomRight = std::optional<Slice2D>{
                    {
                            {*b, *b + 1},
                            {*r, *r + 1}
                    }};
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
//void applyTileMappingsForTensors(Tensor &t1, Tensor &t2, Graph &graph, const GridPartitioning &tileMappings) {
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
//GridPartitioning getTensorRowMappingToTiles(unsigned numRows, unsigned numTilesToUse) {
//    unsigned currentRow = 0;
//
//    GridPartitioning result;
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
//void printTileMappings(const GridPartitioning &tileMappings) {
//    for (const auto &tileMappings : tileMappings) {
//        const auto tile = tileMappings.first;
//        const auto fromRow = tileMappings.second.first;
//        const auto toRow = tileMappings.second.second;
//        std::cout << "[" << fromRow << ":" << toRow << "] -> " << tile << std::endl;
//    }
//}

#endif //LBM_GRAPHCORE_STRUCTUREGRIDUTILS_H
