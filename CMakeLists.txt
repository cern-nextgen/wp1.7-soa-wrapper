cmake_minimum_required(VERSION 3.10)

project(soa_wrapper)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(soa_wrapper main.cpp)

target_include_directories(soa_wrapper PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
