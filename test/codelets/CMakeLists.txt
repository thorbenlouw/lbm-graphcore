file(GLOB SRCS *.cpp)
include(GoogleTest)
ADD_EXECUTABLE(test-codelets ${SRCS})

gtest_discover_tests(test-codelets)
TARGET_LINK_LIBRARIES(test-codelets
        poplar
        poplin
        poputil
        libgtest
        libgmock
        )

configure_file(../../main/codelets/D2Q9Codelets.cpp D2Q9Codelets.cpp COPYONLY)