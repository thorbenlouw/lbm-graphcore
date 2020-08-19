set(CMAKE_CXX_STANDARD 17)


include_directories(lib/lodepng)
include_directories(lib/cxxopts)

set(Main_SRCS
        DoubleRoll.hpp
        GraphcoreUtils.hpp
        LbmParams.hpp
        LatticeBoltzmann.hpp
        StructuredGridUtils.hpp
        StencilUtils.hpp
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

add_executable(low_level_stencil
        LowLevelStencil.cpp
        lib/lodepng/lodepng.cpp
        ${Main_SRCS}
        )


add_executable(halo_regions
        HaloRegionApproaches.cpp
        ${Main_SRCS}
        )

add_executable(poplibs_stencil
        PoplibsStencil.cpp
        lib/lodepng/lodepng.cpp
        ${Main_SRCS}
        )

add_executable(cpu_stencil
        CpuStencil.cpp
        lib/lodepng/lodepng.cpp
        ${Main_SRCS}
        )

add_executable(viz_tile_map
        VisualiseTileMapping.cpp
        lib/lodepng/lodepng.cpp
        ${Main_SRCS}
        )

add_executable(tile_mapping_stats
        TileMappingStats.cpp
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
        poputil
        )


target_link_libraries(halo_regions
        poplar
        popops
        poputil
        )

target_link_libraries(lbm_with_kernels
        poplar
        poputil
        poplin
        )


target_link_libraries(poplibs_stencil
        poplar
        poputil
        poplin
        )


target_link_libraries(low_level_stencil
        poplar
        poputil
        )


target_link_libraries(cpu_stencil
        PUBLIC OpenMP::OpenMP_CXX
        )

install(TARGETS lbm DESTINATION bin)

install(TARGETS lbm_with_kernels DESTINATION bin)

configure_file(codelets/D2Q9Codelets.cpp codelets/D2Q9Codelets.cpp COPYONLY)
configure_file(codelets/StencilCodelet.cpp codelets/StencilCodelet.cpp COPYONLY)
configure_file(codelets/HaloRegionApproachesCodelets.cpp codelets/HaloRegionApproachesCodelets.cpp COPYONLY)