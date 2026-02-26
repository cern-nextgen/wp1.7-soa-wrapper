set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

add_executable(test_cpu test_cpu.cpp)
target_include_directories(test_cpu PUBLIC ${CMAKE_SOURCE_DIR}/tests/include ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_cpu PRIVATE memlayout GTest::gtest_main)

gtest_discover_tests(test_cpu)
