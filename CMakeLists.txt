cmake_minimum_required(VERSION 3.14)
project(cmake_and_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# Allow "std::span" on the device
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Avoid linker error "undefined reference to std::__cxx11::basic_string"
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)


# CUDA ----------------------------------------

add_library(kernel STATIC)
target_sources(kernel PRIVATE kernel.cu)
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
set_target_properties(unittests PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES 89)
target_include_directories(unittests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(unittests GTest::gtest_main kernel)

include(GoogleTest)
gtest_discover_tests(unittests)
