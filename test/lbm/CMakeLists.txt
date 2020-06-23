file(GLOB SRCS *.cpp)

ADD_EXECUTABLE(test-lbm ${SRCS})

TARGET_LINK_LIBRARIES(test-lbm
        poplar
        poplin
        poputil
        libgtest
        libgmock
        )

add_test(NAME test-lbm
        COMMAND tets-lbm)