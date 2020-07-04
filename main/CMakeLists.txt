set(CMAKE_CXX_STANDARD 17)

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

target_link_libraries(lbm_with_kernels
        poplar
        poputil
        poplin
        )

install(TARGETS lbm DESTINATION bin)

install(TARGETS lbm_with_kernels DESTINATION bin)

configure_file(D2Q9Codelets.cpp D2Q9Codelets.cpp COPYONLY)