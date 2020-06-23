set(Main_SRCS
        lbm.cpp
        DoubleRoll.hpp
        )


add_executable(lbm
        ${Main_SRCS}
        )


# Specify here the libraries this program depends on
target_link_libraries(lbm
        poplar
        poputil
        poplin
        )

install(TARGETS lbm DESTINATION bin)