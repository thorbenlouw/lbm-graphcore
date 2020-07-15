set(CMAKE_CXX_STANDARD 17)

include_directories(lib/lodepng)
include_directories(lib/cxxopts)

set(Main_SRCS
        DoubleRoll.hpp
        GraphcoreUtils.hpp
        LbmParams.hpp
        LatticeBoltzmann.hpp
        StructuredGridUtils.hpp
        )

add_executable(lbm
        lbm.cpp
        ${Main_SRCS}
        )


add_executable(cpu-lbm
        CpuLbm.cpp
        ${Main_SRCS}
        )

add_executable(lbm_runner
        LbmRunner.cpp
        ${Main_SRCS}
        )

add_executable(stencil
        stencil.cpp
        lib/lodepng/lodepng.cpp
        ${Main_SRCS}
        )

add_executable(lbm_with_kernels
        lbm_with_kernels.cpp
        ${Main_SRCS}
        )

# Specify here the libraries this program depends on
target_link_libraries(lbm
        poplar
        poputil
        poplin
        )

target_link_libraries(lbm_runner
        poplar
        )

target_link_libraries(lbm_with_kernels
        poplar
        poputil
        poplin
        )

target_link_libraries(stencil
        poplar
        poputil
        )

install(TARGETS lbm DESTINATION bin)

install(TARGETS lbm_with_kernels DESTINATION bin)

configure_file(D2Q9Codelets.cpp D2Q9Codelets.cpp COPYONLY)