cmake_minimum_required(VERSION 3.05)
project(lbm_graphcore)

find_package(poplar REQUIRED)
#find_package(poputil REQUIRED)
#find_package(popops REQUIRED)
#find_package(poplibs REQUIRED)

set(CMAKE_CXX_STANDARD 14)
enable_testing()


add_subdirectory(main)
add_subdirectory(test)