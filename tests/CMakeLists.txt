include(FetchContent)
FetchContent_Declare(
    googletest
    DOWNLOAD_EXTRACT_TIMESTAMP NEW
    URL https://github.com/google/googletest/archive/52eb8108c5bdec04579160ae17225d66034bd723.zip
)
FetchContent_MakeAvailable(googletest)
enable_testing()
include(GoogleTest)

if(ENABLE_CPU)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    add_executable(test_cpu test_cpu.cpp)
    target_include_directories(test_cpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(test_cpu PRIVATE memlayout GTest::gtest_main)
    gtest_discover_tests(test_cpu)
endif()

if(ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    enable_language(CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    add_executable(test_cuda test_cuda.cu)
    target_include_directories(test_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(test_cuda PRIVATE memlayout GTest::gtest_main)
    set_target_properties(test_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES native)
    gtest_discover_tests(test_cuda)
endif()