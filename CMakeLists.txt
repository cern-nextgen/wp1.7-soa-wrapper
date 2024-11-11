cmake_minimum_required(VERSION 3.10)

project(wrapper)

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(wrapper main.cpp)

target_include_directories(wrapper PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
