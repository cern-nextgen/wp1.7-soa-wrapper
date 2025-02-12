#include "allocator.h"
#include "debug.h"
#include "factory.h"
#include "wrapper.h"

#include <iostream>
#include <span>
#include <string>

#include <gtest/gtest.h>


struct Point2D { double x, y; };

template <template <class> class F>
struct S {
    F<int> x;
    F<int> y;
    F<Point2D> point;
    F<double> identifier;

    int abs2() const { return x * x + y * y; }
    int& getX() { return x; }
    const int& getX() const { return x; }
    void setX(int x_new) { x = x_new; }
};

template <class L, class R>
bool operator==(L l, R r) { return l.x == r.x && l.y == r.y && l.point == r.point && l.identifier == r.identifier; }
bool operator==(Point2D l, Point2D r) { return l.x == r.x && l.y == r.y; }

template <class T>
using my_span = std::span<T>;  // avoid clang bug about default template parameters

template <class T>
using pointer_type = T*;

template<template <class> class F, wrapper::layout L>
void initialize(std::size_t N, wrapper::wrapper<F, S, L> &w) {
    for (int i = 0; i < N; ++i) {
        S<wrapper::reference> r = w[i];
        w[i].x = i - 10;
        w[i].y = i + 50;
        w[i].point = {0.5 * i, 0.5 * i};
        w[i].identifier = 0.1 * i;
    }
}

template<template <class> class F, wrapper::layout L>
void assert_equal_to_initialization(std::size_t N, const wrapper::wrapper<F, S, L> &w) {
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(w[i].x, i - 10);
        EXPECT_EQ(w[i].y, i + 50);
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