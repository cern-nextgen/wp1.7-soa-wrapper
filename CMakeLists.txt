cmake_minimum_required(VERSION 4.0)
project(memlayout LANGUAGES CXX)

option(ENABLE_CPU "Build CPU tests" ON)
option(ENABLE_CUDA "Build CUDA tests" ON)

add_library(memlayout INTERFACE)
target_include_directories(memlayout INTERFACE include)

add_subdirectory(tests)
