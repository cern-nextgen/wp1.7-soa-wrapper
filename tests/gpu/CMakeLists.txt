set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
enable_language(CUDA)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

add_executable(test_cuda test_cuda.cu)
target_include_directories(test_cuda PUBLIC ${CMAKE_SOURCE_DIR}/tests/include)
target_link_libraries(test_cuda PRIVATE memlayout::Wrapper GTest::gtest_main)
set_target_properties(test_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES native)

gtest_discover_tests(test_cuda)
