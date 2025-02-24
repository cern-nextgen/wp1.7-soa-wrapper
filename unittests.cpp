#include "allocator.h"
#include "debug.h"
#include "factory.h"
#include "kernel.h"
#include "skeleton.h"
#include "wrapper.h"

#include <iostream>
#include <span>

#include <gtest/gtest.h>

template <class L, class R>
bool operator==(L l, R r) { return l.x == r.x && l.y == r.y && l.point == r.point && l.identifier == r.identifier; }
bool operator==(Point2D l, Point2D r) { return l.x == r.x && l.y == r.y; }

template <class T>
using my_span = std::span<T>;  // avoid clang error about default template parameters

template <class T>
using pointer_type = T*;

template<template <class> class F, wrapper::layout L>
void initialize(std::size_t N, wrapper::wrapper<F, S, L> &w) {
    for (int i = 0; i < N; ++i) {
        S<wrapper::reference> r = w[i];
        w[i].x = i - 10;
        w[i].y = i + 10;
        w[i].point = {0.5 * i, 0.5 * i};
        w[i].identifier = 0.1 * i;
    }
}

template<template <class> class F, wrapper::layout L>
void assert_equal_to_initialization(std::size_t N, const wrapper::wrapper<F, S, L> &w) {
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(w[i].x, i - 10);
        EXPECT_EQ(w[i].y, i + 10);
        EXPECT_EQ(w[i].point, Point2D(0.5 * i, 0.5 * i));
        EXPECT_EQ(w[i].identifier, 0.1 * i);
    }
}

template<template <class> class F, wrapper::layout L>
void test_random_access(std::size_t N, wrapper::wrapper<F, S, L> w) {

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
    debug::counters expected_count = {0, 1, 0, 0, 1, 0, 2};
    {
        std::size_t N = 18;
        wrapper::wrapper<debug::vector, S, wrapper::layout::aos> w{
            debug::vector<S<wrapper::value>>(N)
        };
        test_random_access(N, std::move(w));
    }
    EXPECT_EQ(expected_count, debug::call_counter::count);
}
TEST(Wrapper, SoA) {
    debug::call_counter::count.reset();
    debug::counters expected_count = {0, 4, 0, 0, 4, 0, 8};
    {
        std::size_t N = 18;
        wrapper::wrapper<debug::vector, S, wrapper::layout::soa> w{
            S<debug::vector>{
                debug::vector<int>(N),
                debug::vector<int>(N),
                debug::vector<Point2D>(N),
                debug::vector<double>(N)
            }
        };
        test_random_access(N, std::move(w));
    }
    EXPECT_EQ(expected_count, debug::call_counter::count);
}

TEST(SpanWrapper, AoS) {
    debug::call_counter::count.reset();
    debug::counters expected_count = {0, 1, 0, 0, 0, 0, 1};
    {
        std::size_t N = 18;
        debug::vector<S<wrapper::value>> data(N);
        wrapper::wrapper<my_span, S, wrapper::layout::aos> w{ data };
        test_random_access(N, std::move(w));
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
        wrapper::wrapper<my_span, S, wrapper::layout::soa> w{{ x, y, points, identifier }};
        test_random_access(N, std::move(w));
    }
    EXPECT_EQ(expected_count, debug::call_counter::count);
}

TEST(DefaultWrapper, AoS) {
    debug::call_counter::count.reset();
    debug::counters expected_count = {0, 1, 0, 0, 1, 0, 2};
    {
        std::size_t N = 5;
        auto w = factory::default_wrapper<debug::vector, S, wrapper::layout::aos>(N);
        test_random_access(N, std::move(w));
    }
    EXPECT_EQ(expected_count, debug::call_counter::count);
}
TEST(DefaultWrapper, SoA) {
    debug::call_counter::count.reset();
    debug::counters expected_count = {0, 4, 0, 0, 4, 0, 8};
    {
        std::size_t N = 5;
        auto w = factory::default_wrapper<debug::vector, S, wrapper::layout::soa>(N);
        test_random_access(N, std::move(w));
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
        test_random_access(N, std::move(w));
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
        test_random_access(N, std::move(w));
    }
    EXPECT_EQ(expected_count, debug::call_counter::count);
}

TEST(PointerWrapper, AoS) {
    constexpr std::size_t N = 18;
    S<wrapper::value> data[N];
    wrapper::wrapper<pointer_type, S, wrapper::layout::aos> w = { data };
    test_random_access(N, std::move(w));
}
TEST(PointerWrapper, SoA) {
    constexpr std::size_t N = 18;
    int x[N];
    int y[N];
    Point2D point[N];
    double identifier[N];
    wrapper::wrapper<pointer_type, S, wrapper::layout::soa> w = { x, y, point, identifier };
    test_random_access(N, std::move(w));
}

TEST(UnifiedMemoryWrapper, AoS) {
    constexpr std::size_t N = 18;
    kernel::UnifiedMemoryManager<S, wrapper::layout::aos> umm(N);
    auto w = umm.create_wrapper();
    test_random_access(N, w);
    kernel::apply(N, w);
    for (int i = 0; i < N; ++i) EXPECT_EQ(w[i].y, 2 * i);
}
TEST(UnifiedMemoryWrapper, SoA) {
    constexpr std::size_t N = 18;
    kernel::UnifiedMemoryManager<S, wrapper::layout::soa> umm(N);
    auto w = umm.create_wrapper();
    test_random_access(N, w);
    kernel::apply(N, w);
    for (int i = 0; i < N; ++i) EXPECT_EQ(w[i].y, 2 * i);
}

TEST(DeviceSpanWrapper, AoS) {
    constexpr std::size_t N = 18;
    constexpr std::size_t size = N * sizeof(S<wrapper::value>);

    auto * h_data = (S<wrapper::value> *)malloc(size);
    wrapper::wrapper<kernel::pointer_type, S, wrapper::layout::aos> h_w{h_data};
    test_random_access(N, h_w);

    S<wrapper::value> *d_data;
    kernel::cuda_malloc((void **) &d_data, size);
    wrapper::wrapper<kernel::span_type, S, wrapper::layout::aos> d_w{{d_data, d_data + size}};
    kernel::cuda_memcpy(d_data, h_data, size, kernel::copy_flag::cudaMemcpyHostToDevice);
    kernel::apply(N, d_w);
    kernel::cuda_memcpy(h_data, d_data, size, kernel::copy_flag::cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) EXPECT_EQ(h_w[i].y, 2 * i);

    kernel::cuda_free(d_data);
    free(h_data);
}

TEST(DeviceSpanWrapper, SoA) {
    constexpr std::size_t N = 18;

    auto * h_x = (int *)malloc(N * sizeof(int));
    auto * h_y = (int *)malloc(N * sizeof(int));
    auto * h_point = (Point2D *)malloc(N * sizeof(Point2D));
    auto * h_identifier = (double *)malloc(N * sizeof(double));
    wrapper::wrapper<kernel::pointer_type, S, wrapper::layout::soa> h_w{{h_x, h_y, h_point, h_identifier}};
    test_random_access(N, h_w);

    int * d_x; kernel::cuda_malloc((void **) &d_x, N * sizeof(int)); kernel::cuda_memcpy(d_x, h_x, N * sizeof(int), kernel::copy_flag::cudaMemcpyHostToDevice);
    int * d_y; kernel::cuda_malloc((void **) &d_y, N * sizeof(int)); kernel::cuda_memcpy(d_y, h_y, N * sizeof(int), kernel::copy_flag::cudaMemcpyHostToDevice);
    Point2D * d_point; kernel::cuda_malloc((void **) &d_point, N * sizeof(Point2D)); kernel::cuda_memcpy(d_point, h_point, N * sizeof(Point2D), kernel::copy_flag::cudaMemcpyHostToDevice);
    double * d_identifier; kernel::cuda_malloc((void **) &d_identifier, N * sizeof(double)); kernel::cuda_memcpy(d_identifier, h_identifier, N * sizeof(double), kernel::copy_flag::cudaMemcpyHostToDevice);

    wrapper::wrapper<kernel::span_type, S, wrapper::layout::soa> d_w{{
        {d_x, d_x + N * sizeof(int)},
        {d_y, d_y + N * sizeof(int)},
        {d_point, d_point + N * sizeof(Point2D)},
        {d_identifier, d_identifier + N * sizeof(double)}
    }};

    kernel::apply(N, d_w);
    kernel::cuda_memcpy(h_x, d_x, N * sizeof(int), kernel::copy_flag::cudaMemcpyDeviceToHost);
    kernel::cuda_memcpy(h_y, d_y, N * sizeof(int), kernel::copy_flag::cudaMemcpyDeviceToHost);
    kernel::cuda_memcpy(h_point, d_point, N * sizeof(Point2D), kernel::copy_flag::cudaMemcpyDeviceToHost);
    kernel::cuda_memcpy(h_identifier, d_identifier, N * sizeof(double), kernel::copy_flag::cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) EXPECT_EQ(h_w[i].y, 2 * i);

    kernel::cuda_free(d_x);
    kernel::cuda_free(d_y);
    kernel::cuda_free(d_point);
    kernel::cuda_free(d_identifier);

    free(h_x);
    free(h_y);
    free(h_point);
    free(h_identifier);
}