file(GLOB SRCS *.cpp)
include(GoogleTest)
ADD_EXECUTABLE(test-lbm ${SRCS})

gtest_discover_tests(test-lbm)
TARGET_LINK_LIBRARIES(test-lbm
        poplar
        poplin
        poputil
        libgtest
        libgmock
        )