cmake_minimum_required(VERSION 3.14)
project(cmake_and_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED True)


# CUDA ----------------------------------------

add_library(kernel STATIC kernel.cu kernel.h)
set_target_properties(kernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES 89)
target_include_directories(kernel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


# Executable ----------------------------------

add_executable(wrapper main.cpp)
set_target_properties(wrapper PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES 89)
target_include_directories(wrapper PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(wrapper PRIVATE kernel)

# Unit Tests ----------------------------------

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    DOWNLOAD_EXTRACT_TIMESTAMP NEW
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(unittests unittests.cpp)
target_include_directories(unittests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(unittests GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(unittests)
