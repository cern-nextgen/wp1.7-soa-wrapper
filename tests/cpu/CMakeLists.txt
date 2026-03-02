add_executable(test_cpu test_cpu.cpp)
target_include_directories(test_cpu PUBLIC ${CMAKE_SOURCE_DIR}/tests/include)
target_link_libraries(test_cpu PRIVATE memlayout::Wrapper GTest::gtest_main)

gtest_discover_tests(test_cpu)
