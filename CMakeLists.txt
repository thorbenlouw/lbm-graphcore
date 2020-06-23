cmake_minimum_required(VERSION 3.05)
project(lbm_graphcore)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -ansi -Wno-deprecated")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -ansi -Wno-deprecated")
endif()

find_package(poplar REQUIRED)
find_package(poplibs REQUIRED)

enable_testing()

add_subdirectory(main)
add_subdirectory(test)