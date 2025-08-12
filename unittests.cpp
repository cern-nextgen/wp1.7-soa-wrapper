#include "allocator.h"
#include "debug.h"
#include "factory.h"
#include "kernel.h"
#include "skeleton.h"
#include "wrapper.h"

#include <iostream>
#include <memory>
#include <span>
#include <vector>

#include <gtest/gtest.h>

template <class L, class R>
bool operator==(L l, R r) { return l.x == r.x && l.y == r.y && l.point == r.point && l.identifier == r.identifier; }
bool operator==(Point2D l, Point2D r) { return l.x == r.x && l.y == r.y; }

template <class T>
using pointer_type = T*;

template<template <class> class F, wrapper::layout L>
void initialize(std::size_t N, wrapper::wrapper<S, F, L> &w) {
    for (int i = 0; i < N; ++i) {
        S<wrapper::reference> r = w[i];
        w[i].x = i - 10;
        w[i].y = i + 10;
        w[i].point = {0.5 * i, 0.5 * i};
        w[i].identifier = 0.1 * i;
    }
}

template<template <class> class F, wrapper::layout L>
void assert_equal_to_initialization(std::size_t N, const wrapper::wrapper<S, F, L> &w) {
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(w[i].x, i - 10);
        EXPECT_EQ(w[i].y, i + 10);
        EXPECT_EQ(w[i].point, Point2D(0.5 * i, 0.5 * i));
        EXPECT_EQ(w[i].identifier, 0.1 * i);
    }
}

template<template <class> class F, wrapper::layout L>
void test_random_access(std::size_t N, wrapper::wrapper<S, F, L> &w) {

    initialize(N, w);
    assert_equal_to_initialization(N, w);

    // const_reference
    for (int i = 0; i < N; ++i) {
        S<wrapper::const_reference> cr = w[i];
        EXPECT_EQ(cr.abs2(), w[i].abs2());
    }
    assert_equal_to_initialization(N, w);

    // value
    for (int i = 0; i < N; ++i) {
        S<wrapper::value> v = w[i];

        v.setX(i - 1);
        v.y = i + 5;
        v.point = {5.0 * i, 5.0 * i};
        v.identifier = 0.1 * i;

        EXPECT_NE(v.getX(), w[i].x);
        EXPECT_NE(v.abs2(), w[i].abs2());
    }
    assert_equal_to_initialization(N, w);
}

TEST(Wrapper, AoS) {
    debug::call_counter::count.reset();
    debug::counters expected_count = {0, 1, 0, 0, 0, 0, 1};
    {
        std::size_t N = 18;
        wrapper::wrapper<S, debug::vector, wrapper::layout::aos> w{
            debug::vector<S<wrapper::value>>(N)
        };
        test_random_access(N, w);
    }
    EXPECT_EQ(expected_count, debug::call_counter::count);
}
TEST(Wrapper, SoA) {
    debug::call_counter::count.reset();
    debug::counters expected_count = {0, 4, 0, 0, 0, 0, 4};
    {
        std::size_t N = 18;
        wrapper::wrapper<S, debug::vector, wrapper::layout::soa> w{
            S<debug::vector>{
                debug::vector<int>(N),
                debug::vector<int>(N),
                debug::vector<Point2D>(N),
                debug::vector<double>(N)
            }
        };
        test_random_access(N, w);
    }
    EXPECT_EQ(expected_count, debug::call_counter::count);
}

TEST(SpanWrapper, AoS) {
    debug::call_counter::count.reset();
    debug::counters expected_count = {0, 1, 0, 0, 0, 0, 1};
    {
        std::size_t N = 18;
        debug::vector<S<wrapper::value>> data(N);
        wrapper::wrapper<S, std::span, wrapper::layout::aos> w{ data };
        test_random_access(N, w);
    }
    EXPECT_EQ(expected_count, debug::call_counter::count);
}
TEST(SpanWrapper, SoA) {
    debug::call_counter::count.reset();
    debug::counters expected_count = {0, 4, 0, 0, 0, 0, 4};
    {
        std::size_t N = 18;
        debug::vector<int> x(N);
        debug::vector<int> y(N);
        debug::vector<Point2D> points(N);
        debug::vector<double> identifier(N);
        wrapper::wrapper<S, std::span, wrapper::layout::soa> w{{ x, y, points, identifier }};
        test_random_access(N, w);
    }
    EXPECT_EQ(expected_count, debug::call_counter::count);
}

TEST(DefaultWrapper, AoS) {
    debug::call_counter::count.reset();
    debug::counters expected_count = {0, 1, 0, 0, 0, 0, 1};
    {
        std::size_t N = 5;
        auto w = factory::default_wrapper<S, debug::vector, wrapper::layout::aos>(N);
        test_random_access(N, w);
    }
    EXPECT_EQ(expected_count, debug::call_counter::count);
}
TEST(DefaultWrapper, SoA) {
    debug::call_counter::count.reset();
    debug::counters expected_count = {4, 4, 0, 0, 0, 0, 8};  // {0, 4, 0, 0, 4, 0, 8}
    {
        std::size_t N = 5;
        auto w = factory::default_wrapper<S, debug::vector, wrapper::layout::soa>(N);
        test_random_access(N, w);
    }
    EXPECT_EQ(expected_count, debug::call_counter::count);
}

TEST(BufferWrapper, AoS) {
    debug::call_counter::count.reset();
    debug::counters expected_count = {0, 0, 0, 0, 0, 0, 0};
    {
        std::size_t N = 18;
        std::size_t bytes = 1024;
        char buffer[bytes];
        auto w = factory::buffer_wrapper<S, wrapper::layout::aos>(buffer, bytes);
        test_random_access(N, w);
    }
    EXPECT_EQ(expected_count, debug::call_counter::count);
}
TEST(BufferWrapper, SoA) {
    debug::call_counter::count.reset();
    debug::counters expected_count = {0, 0, 0, 0, 0, 0, 0};
    {
        std::size_t N = 18;
        std::size_t bytes = 1024;
        char buffer[bytes];
        auto w = factory::buffer_wrapper<S, wrapper::layout::soa>(buffer, bytes);
        test_random_access(N, w);
    }
    EXPECT_EQ(expected_count, debug::call_counter::count);
}

TEST(PointerWrapper, AoS) {
    constexpr std::size_t N = 18;
    S<wrapper::value> data[N];
    wrapper::wrapper<S, pointer_type, wrapper::layout::aos> w = { data };
    test_random_access(N, w);
}
TEST(PointerWrapper, SoA) {
    constexpr std::size_t N = 18;
    int x[N];
    int y[N];
    Point2D point[N];
    double identifier[N];
    wrapper::wrapper<S, pointer_type, wrapper::layout::soa> w = { x, y, point, identifier };
    test_random_access(N, w);
}

/*template <class T>
using managed_memory_vector = std::vector<T, kernel::ManagedMemoryAllocator<T>>;

TEST(ManagedMemorySpanWrapper, AoS) {
    constexpr std::size_t N = 18;
    wrapper::wrapper<S, managed_memory_vector, wrapper::layout::aos> w{
        managed_memory_vector<S<wrapper::value>>(N)
    };
    initialize(N, w);
    assert_equal_to_initialization(N, w);

    wrapper::wrapper<S, kernel::span_type, wrapper::layout::aos> w_span(w);
    int cuda_error = kernel::apply(N, w_span);
    EXPECT_EQ(cuda_error, 0);

    for (int i = 0; i < N; ++i) EXPECT_EQ(w_span[i].y, 2 * i);
}
TEST(ManagedMemorySpanWrapper, SoA) {
    constexpr std::size_t N = 18;
    wrapper::wrapper<managed_memory_vector, S, wrapper::layout::soa> w{{
        managed_memory_vector<int>(N),
        managed_memory_vector<int>(N),
        managed_memory_vector<Point2D>(N),
        managed_memory_vector<double>(N)
    }};
    initialize(N, w);
    assert_equal_to_initialization(N, w);

    wrapper::wrapper<S, kernel::span_type, wrapper::layout::soa> w_span(w);
    int cuda_error = kernel::apply(N, w_span);
    EXPECT_EQ(cuda_error, 0);

    for (int i = 0; i < N; ++i) EXPECT_EQ(w_span[i].y, 2 * i);
}

TEST(DeviceSpanWrapper, AoS) {
    constexpr std::size_t N = 18;

    wrapper::wrapper<S, debug::vector, wrapper::layout::aos> h_w{{N}};
    initialize(N, h_w);
    assert_equal_to_initialization(N, h_w);
    wrapper::wrapper<S, kernel::span_type, wrapper::layout::aos> h_span(h_w);

    wrapper::wrapper<kernel::device_memory_array, S, wrapper::layout::aos> d_w{N};
    wrapper::wrapper<std::span, S, wrapper::layout::aos> d_span(d_w);
    kernel::cuda_memcpy(d_span, h_span, N, kernel::cuda_memcpy_kind::cudaMemcpyHostToDevice);
    int cuda_error = kernel::apply(N, d_span);
    EXPECT_EQ(cuda_error, 0);
    kernel::cuda_memcpy(h_span, d_span, N, kernel::cuda_memcpy_kind::cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) EXPECT_EQ(h_span[i].y, 2 * i);
}
TEST(DeviceSpanWrapper, SoA) {
    constexpr std::size_t N = 18;

    wrapper::wrapper<S, debug::vector, wrapper::layout::soa> h_w{{ N, N, N, N }};
    initialize(N, h_w);
    assert_equal_to_initialization(N, h_w);
    wrapper::wrapper<S, kernel::span_type, wrapper::layout::soa> h_span(h_w);

    wrapper::wrapper<S, kernel::device_memory_array, wrapper::layout::soa> d_w{{ N, N, N, N }};
    wrapper::wrapper<S, kernel::span_type, wrapper::layout::soa> d_span(d_w);
    kernel::cuda_memcpy(d_span, h_span, N, kernel::cuda_memcpy_kind::cudaMemcpyHostToDevice);
    int cuda_error = kernel::apply(N, d_span);
    EXPECT_EQ(cuda_error, 0);
    kernel::cuda_memcpy(h_span, d_span, N, kernel::cuda_memcpy_kind::cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) EXPECT_EQ(h_span[i].y, 2 * i);
}*/