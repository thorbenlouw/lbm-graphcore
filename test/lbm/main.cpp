#include "gtest/gtest.h"
#include "../../main/DoubleRoll.hpp"

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}

typedef struct {
    size_t src_xFrom, src_xTo, src_yFrom, src_yTo, dst_xFrom, dst_xTo, dst_yFrom, dst_yTo;
} UnpackedRollSlices;

auto unpack(SliceRegion src, SliceRegion dst) -> UnpackedRollSlices {
    auto const &[src_slices, dst_slices] = determineSrcAndDstSlices({3, 3}, {0, 0});
    auto const &[src_xSlice, src_ySlice] = src;
    auto const &[src_xFrom, src_xTo] = src_xSlice;
    auto const &[src_yFrom, src_yTo] = src_ySlice;
    auto const &[dst_xSlice, dst_ySlice] = dst;
    auto const &[dst_xFrom, dst_xTo] = dst_xSlice;
    auto const &[dst_yFrom, dst_yTo] = dst_ySlice;
    return {
            .src_xFrom = src_xFrom,
            .src_xTo = src_xTo,
            .src_yFrom = src_yFrom,
            .src_yTo = src_yTo,
            .dst_xFrom = dst_xFrom,
            .dst_xTo = dst_xTo,
            .dst_yFrom = dst_yFrom,
            .dst_yTo = dst_yTo,
    };
}

auto assertMiddleBlockIs(std::vector<SliceRegion> x, size_t xFrom, size_t xTo, size_t yFrom, size_t yTo) {


}

enum RegionName {
    Middle,
    Left,
    Right,
    Top,
    Bottom,
    TopLeft,
    BottomLeft,
    TopRight,
    BottomRight
};

const auto RegionNameStrings = std::array{"Middle",
                                          "Left",
                                          "Right",
                                          "Top",
                                          "Bottom",
                                          "TopLeft",
                                          "BottomLeft",
                                          "TopRight",
                                          "BottomRight"};


class To {
public:

    const SliceRegion &actualSrc;
    const SliceRegion &actualDst;
    const SliceRegion &expectedSrc;
    const std::string description;

    To(const SliceRegion &actualSrc, const SliceRegion &actualDst, const SliceRegion &expectedSrc,
       const std::string &descr) : actualSrc(actualSrc),
                                   actualDst(actualDst),
                                   expectedSrc(expectedSrc),
                                   description(descr) {

    }

    auto to(const SliceRegion &expectedDst) -> void {
        ASSERT_EQ(expectedSrc, actualSrc) << " " << description;
        ASSERT_EQ(expectedDst, actualDst) << " " << description;
    }
};

class Copies {
public:
    const SliceRegion &actualSrc;
    const SliceRegion &actualDst;
    const std::string descr;

    Copies(const SliceRegion &actualSrc, const SliceRegion &actualDst, const std::string &descr) : actualSrc(actualSrc),
                                                                                                   actualDst(actualDst),
                                                                                                   descr(descr) {};

    To isCopiedFrom(const SliceRegion &expectedSrc) {
        return To{actualSrc, actualDst, expectedSrc, descr};
    }
};

class ThatRegion {
public:
    const std::vector<SliceRegion> &src;
    const std::vector<SliceRegion> &dst;

    ThatRegion(const std::vector<SliceRegion> &src, const std::vector<SliceRegion> &dst) : src(src), dst(dst) {
    }

    Copies thatRegion(const RegionName region) {
        return Copies{src[region], dst[region], RegionNameStrings[region]};
    }
};

auto assertUsing(const std::vector<SliceRegion> &src, const std::vector<SliceRegion> &dst) -> ThatRegion {
    return ThatRegion{src, dst};
}

TEST(testDoubleRoll_determineSrcAndDstSlices, noOffset) {
    auto const &[src_slices, dst_slices] = determineSrcAndDstSlices({3, 4}, {0, 0});

    assertUsing(src_slices, dst_slices)
            .thatRegion(Middle)
            .isCopiedFrom({{1, 2},
                           {1, 3}})
            .to({{1, 2},
                 {1, 3}});
    assertUsing(src_slices, dst_slices)
            .thatRegion(Top)
            .isCopiedFrom({{1, 2},
                           {0, 1}})
            .to({{1, 2},
                 {0, 1}});
    assertUsing(src_slices, dst_slices)
            .thatRegion(Left)
            .isCopiedFrom({{0, 1},
                           {1, 3}})
            .to({{0, 1},
                 {1, 3}});
    assertUsing(src_slices, dst_slices)
            .thatRegion(Right)
            .isCopiedFrom({{2, 3},
                           {1, 3}})
            .to({{2, 3},
                 {1, 3}});
    assertUsing(src_slices, dst_slices)
            .thatRegion(Bottom)
            .isCopiedFrom({{1, 2},
                           {3, 4}})
            .to({{1, 2},
                 {3, 4}});
    assertUsing(src_slices, dst_slices)
            .thatRegion(TopLeft)
            .isCopiedFrom({{0, 1},
                           {0, 1}})
            .to({{0, 1},
                 {0, 1}});
    assertUsing(src_slices, dst_slices)
            .thatRegion(TopRight)
            .isCopiedFrom({{2, 3},
                           {0, 1}})
            .to({{2, 3},
                 {0, 1}});
    assertUsing(src_slices, dst_slices)
            .thatRegion(BottomLeft)
            .isCopiedFrom({{0, 1},
                           {3, 4}})
            .to({{0, 1},
                 {3, 4}});
    assertUsing(src_slices, dst_slices)
            .thatRegion(BottomRight)
            .isCopiedFrom({{2, 3},
                           {3, 4}})
            .to({{2, 3},
                 {3, 4}});
}


TEST(testDoubleRoll_determineSrcAndDstSlices, rollUp) {
    auto const &[src_slices, dst_slices] = determineSrcAndDstSlices({3, 4}, {0, 1});

    assertUsing(src_slices, dst_slices)
            .thatRegion(Middle)
            .isCopiedFrom({{1, 2},
                           {1, 3}})
            .to({{1, 2},
                 {0, 2}});
    assertUsing(src_slices, dst_slices)
            .thatRegion(Top)
            .isCopiedFrom({{1, 2},
                           {0, 1}})
            .to({{1, 2},
                 {3, 4}});
    assertUsing(src_slices, dst_slices)
            .thatRegion(Left)
            .isCopiedFrom({{0, 1},
                           {1, 3}})
            .to({{0, 1},
                 {0, 2}});
    assertUsing(src_slices, dst_slices)
            .thatRegion(Right)
            .isCopiedFrom({{2, 3},
                           {1, 3}})
            .to({{2, 3},
                 {0, 2}});
    assertUsing(src_slices, dst_slices)
            .thatRegion(Bottom)
            .isCopiedFrom({{1, 2},
                           {3, 4}})
            .to({{1, 2},
                 {2, 3}});
    assertUsing(src_slices, dst_slices)
            .thatRegion(TopLeft)
            .isCopiedFrom({{0, 1},
                           {0, 1}})
            .to({{0, 1},
                 {3, 4}});
    assertUsing(src_slices, dst_slices)
            .thatRegion(TopRight)
            .isCopiedFrom({{2, 3},
                           {0, 1}})
            .to({{2, 3},
                 {3, 4}});
    assertUsing(src_slices, dst_slices)
            .thatRegion(BottomLeft)
            .isCopiedFrom({{0, 1},
                           {3, 4}})
            .to({{0, 1},
                 {2, 3}});
    assertUsing(src_slices, dst_slices)
            .thatRegion(BottomRight)
            .isCopiedFrom({{2, 3},
                           {3, 4}})
            .to({{2, 3},
                 {2, 3}});
}

TEST(testDoubleRoll_determineSrcAndDstSlices, rollRight) {
    auto const &[src_slices, dst_slices] = determineSrcAndDstSlices({3, 4}, {-1, 0});

    assertUsing(src_slices, dst_slices)
            .thatRegion(Middle)
            .isCopiedFrom({{1, 2},
                           {1, 3}})
            .to({{2, 3},
                 {1, 3}});
    assertUsing(src_slices, dst_slices)
            .thatRegion(Top)
            .isCopiedFrom({{1, 2},
                           {0, 1}})
            .to({{2, 3},
                 {0, 1}});
    assertUsing(src_slices, dst_slices)
            .thatRegion(Left)
            .isCopiedFrom({{0, 1},
                           {1, 3}})
            .to({{1, 2},
                 {1, 3}});
    assertUsing(src_slices, dst_slices)
            .thatRegion(Right)
            .isCopiedFrom({{2, 3},
                           {1, 3}})
            .to({{0, 1},
                 {1, 3}});
    assertUsing(src_slices, dst_slices)
            .thatRegion(Bottom)
            .isCopiedFrom({{1, 2},
                           {3, 4}})
            .to({{2, 3},
                 {3, 4}});
    assertUsing(src_slices, dst_slices)
            .thatRegion(TopLeft)
            .isCopiedFrom({{0, 1},
                           {0, 1}})
            .to({{1, 2},
                 {0, 1}});
    assertUsing(src_slices, dst_slices)
            .thatRegion(TopRight)
            .isCopiedFrom({{2, 3},
                           {0, 1}})
            .to({{0, 1},
                 {0, 1}});
    assertUsing(src_slices, dst_slices)
            .thatRegion(BottomLeft)
            .isCopiedFrom({{0, 1},
                           {3, 4}})
            .to({{1, 2},
                 {3, 4}});
    assertUsing(src_slices, dst_slices)
            .thatRegion(BottomRight)
            .isCopiedFrom({{2, 3},
                           {3, 4}})
            .to({{0, 1},
                 {3, 4}});
}
