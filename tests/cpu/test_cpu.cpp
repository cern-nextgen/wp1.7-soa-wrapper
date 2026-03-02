#include <vector>
#include <span>

#include <gtest/gtest.h>

#include "structs.h"
#include "vector.h"
#include "memlayout/wrapper.h"

template<template <class> class Container, memlayout::Layout L>
void initialize(std::size_t N, memlayout::Wrapper<test::Point3D, Container, L> &w) {
    for (int32_t i = 0; i < N; ++i) {
        memlayout::Wrapper<test::Point3D, memlayout::reference> r = w[i];
        w[i].point2d.x = i - 2;
        w[i].point2d.y = i + 2;
        w[i].z = 3 * i;
    }
}

template<template <class> class Container, memlayout::Layout L>
void assert_equal_to_initialization(std::size_t N, const memlayout::Wrapper<test::Point3D, Container, L> &w) {
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(w[i].point2d.x, i - 2);
        EXPECT_EQ(w[i].point2d.y, i + 2);
        EXPECT_EQ(w[i].z, 3 * i);
    }
}

template<template <class> class Container, memlayout::Layout L>
void test_random_access(std::size_t N, memlayout::Wrapper<test::Point3D, Container, L> &w) {

    initialize(N, w);
    assert_equal_to_initialization(N, w);

    // const_reference
    for (int i = 0; i < N; ++i) {
        memlayout::Wrapper<test::Point3D, memlayout::reference> cr = w[i];
        EXPECT_EQ(cr.abs2(), w[i].abs2());
    }
    assert_equal_to_initialization(N, w);

    // value
    for (int32_t i = 0; i < N; ++i) {
        memlayout::Wrapper<test::Point3D, memlayout::value> v = w[i];

        v.point2d = test::Point2D<memlayout::value>{i - 3, i + 3};
        v.setZ(3 * i + 1);

        EXPECT_NE(v.getZ(), w[i].z);
        EXPECT_NE(v.abs2(), w[i].abs2());
    }
    assert_equal_to_initialization(N, w);
}

TEST(Wrapper, AoS) {
    test::call_counter::count.reset();
    test::counters expected_count = {0, 1, 0, 0, 0, 0, 1};
    {
        constexpr std::size_t N = 32;
        memlayout::Wrapper<test::Point3D, test::vector, memlayout::Layout::aos> w{N};
        test_random_access(N, w);
    }
    EXPECT_EQ(expected_count, test::call_counter::count);
}
TEST(Wrapper, SoA) {
    test::call_counter::count.reset();
    test::counters expected_count = {0, 3, 0, 0, 5, 0, 8};
    {
        constexpr std::size_t N = 32;
        memlayout::Wrapper<test::Point3D, test::vector, memlayout::Layout::soa> w{{{{N, N}}, N}};
        test_random_access(N, w);
    }
    EXPECT_EQ(expected_count, test::call_counter::count);
}

TEST(SpanWrapper, AoS) {
    test::call_counter::count.reset();
    test::counters expected_count = {0, 1, 0, 0, 0, 0, 1};
    {
        constexpr std::size_t N = 32;
        test::vector<test::Point3D<memlayout::value>> data(N);
        memlayout::Wrapper<test::Point3D, std::span, memlayout::Layout::aos> w{data};
        test_random_access(N, w);
    }
    EXPECT_EQ(expected_count, test::call_counter::count);
}
TEST(SpanWrapper, SoA) {
    test::call_counter::count.reset();
    test::counters expected_count = {0, 3, 0, 0, 0, 0, 3};
    {
        constexpr std::size_t N = 32;
        test::vector<int32_t> x(N);
        test::vector<int32_t> y(N);
        test::vector<int32_t> z(N);
        memlayout::Wrapper<test::Point3D, std::span, memlayout::Layout::soa> w{{{{x, y}}, z}};
        test_random_access(N, w);
    }
    EXPECT_EQ(expected_count, test::call_counter::count);
}

TEST(PointerWrapper, AoS) {
    constexpr std::size_t N = 32;
    test::Point3D<memlayout::value> data[N];
    memlayout::Wrapper<test::Point3D, memlayout::pointer, memlayout::Layout::aos> w = {data};
    test_random_access(N, w);
}
TEST(PointerWrapper, SoA) {
    constexpr std::size_t N = 32;
    int32_t x[N];
    int32_t y[N];
    int32_t z[N];
    memlayout::Wrapper<test::Point3D, memlayout::pointer, memlayout::Layout::soa> w = {{{{x, y}}, z}};
    test_random_access(N, w);
}

template <memlayout::Layout L>
uint32_t compute_sum(int N, memlayout::Wrapper<test::Point2D, std::span, L> w) {
    uint32_t sum = 0;
    for (int i = 0; i < N; ++i) {
        memlayout::Wrapper<test::Point2D, memlayout::reference> cr = w[i];
        sum += cr.x + cr.y;
    }
    return sum;
}

TEST(OverloadResolution, AoS) {
    memlayout::Wrapper<test::Point2D, std::vector, memlayout::Layout::aos> w{
        {0, 5}, {1, 6}, {2, 7}, {3, 8}, {4, 9}
    };
    uint32_t sum = compute_sum<w.layout_type>(5, w);
    uint32_t expected_sum = 45;
    EXPECT_EQ(expected_sum, sum);
}
TEST(OverloadResolution, SoA) {
    memlayout::Wrapper<test::Point2D, std::vector, memlayout::Layout::soa> w{
        {{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}}
    };
    uint32_t sum = compute_sum<w.layout_type>(5, w);
    uint32_t expected_sum = 45;
    EXPECT_EQ(expected_sum, sum);
}

TEST(VectorToSpan, AoS) {
    memlayout::Wrapper<test::Point2D, std::vector, memlayout::Layout::aos> w{
        {0, 5}, {1, 6}, {2, 7}, {3, 8}, {4, 9}
    };
    memlayout::Wrapper<test::Point2D, std::span, memlayout::Layout::aos> w_span{w};
    uint32_t sum = compute_sum(5, w_span);
    uint32_t expected_sum = 45;
    EXPECT_EQ(expected_sum, sum);
}
TEST(VectorToSpan, SoA) {
    memlayout::Wrapper<test::Point2D, std::vector, memlayout::Layout::soa> w{
        {{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}}
    };
    memlayout::Wrapper<test::Point2D, std::span, memlayout::Layout::soa> w_span{w};
    uint32_t sum = compute_sum(5, w_span);
    uint32_t expected_sum = 45;
    EXPECT_EQ(expected_sum, sum);
}
