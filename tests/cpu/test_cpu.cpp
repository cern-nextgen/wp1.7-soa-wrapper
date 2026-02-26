#include <vector>
#include <span>

#include <gtest/gtest.h>

#include "debug.h"
#include "skeleton.h"
#include "wrapper.h"

template<template <class> class F, wrapper::layout L>
void initialize(std::size_t N, wrapper::wrapper<Skeleton::Point3D, F, L> &w) {
    for (int32_t i = 0; i < N; ++i) {
        wrapper::wrapper<Skeleton::Point3D, wrapper::reference> r = w[i];
        w[i].point2d.x = i - 2;
        w[i].point2d.y = i + 2;
        w[i].z = 3 * i;
    }
}

template<template <class> class F, wrapper::layout L>
void assert_equal_to_initialization(std::size_t N, const wrapper::wrapper<Skeleton::Point3D, F, L> &w) {
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(w[i].point2d.x, i - 2);
        EXPECT_EQ(w[i].point2d.y, i + 2);
        EXPECT_EQ(w[i].z, 3 * i);
    }
}

template<template <class> class F, wrapper::layout L>
void test_random_access(std::size_t N, wrapper::wrapper<Skeleton::Point3D, F, L> &w) {

    initialize(N, w);
    assert_equal_to_initialization(N, w);

    // const_reference
    for (int i = 0; i < N; ++i) {
        wrapper::wrapper<Skeleton::Point3D, wrapper::const_reference> cr = w[i];
        EXPECT_EQ(cr.abs2(), w[i].abs2());
    }
    assert_equal_to_initialization(N, w);

    // value
    for (int32_t i = 0; i < N; ++i) {
        wrapper::wrapper<Skeleton::Point3D, wrapper::value> v = w[i];

        v.point2d = Skeleton::Point2D<wrapper::value>{i - 3, i + 3};
        v.setZ(3 * i + 1);

        EXPECT_NE(v.getZ(), w[i].z);
        EXPECT_NE(v.abs2(), w[i].abs2());
    }
    assert_equal_to_initialization(N, w);
}

TEST(Wrapper, AoS) {
    debug::call_counter::count.reset();
    debug::counters expected_count = {0, 1, 0, 0, 0, 0, 1};
    {
        constexpr std::size_t N = 32;
        wrapper::wrapper<Skeleton::Point3D, debug::vector, wrapper::layout::aos> w{{N}};
        test_random_access(N, w);
    }
    EXPECT_EQ(expected_count, debug::call_counter::count);
}
TEST(Wrapper, SoA) {
    debug::call_counter::count.reset();
    debug::counters expected_count = {0, 3, 0, 0, 5, 0, 8};
    {
        constexpr std::size_t N = 32;
        wrapper::wrapper<Skeleton::Point3D, debug::vector, wrapper::layout::soa> w{{{{N, N}}, N}};
        test_random_access(N, w);
    }
    EXPECT_EQ(expected_count, debug::call_counter::count);
}

TEST(SpanWrapper, AoS) {
    debug::call_counter::count.reset();
    debug::counters expected_count = {0, 1, 0, 0, 0, 0, 1};
    {
        constexpr std::size_t N = 32;
        debug::vector<Skeleton::Point3D<wrapper::value>> data(N);
        wrapper::wrapper<Skeleton::Point3D, std::span, wrapper::layout::aos> w{data};
        test_random_access(N, w);
    }
    EXPECT_EQ(expected_count, debug::call_counter::count);
}
TEST(SpanWrapper, SoA) {
    debug::call_counter::count.reset();
    debug::counters expected_count = {0, 3, 0, 0, 0, 0, 3};
    {
        constexpr std::size_t N = 32;
        debug::vector<int32_t> x(N);
        debug::vector<int32_t> y(N);
        debug::vector<int32_t> z(N);
        wrapper::wrapper<Skeleton::Point3D, std::span, wrapper::layout::soa> w{{{{x, y}}, z}};
        test_random_access(N, w);
    }
    EXPECT_EQ(expected_count, debug::call_counter::count);
}

TEST(PointerWrapper, AoS) {
    constexpr std::size_t N = 32;
    Skeleton::Point3D<wrapper::value> data[N];
    wrapper::wrapper<Skeleton::Point3D, wrapper::pointer, wrapper::layout::aos> w = {data};
    test_random_access(N, w);
}
TEST(PointerWrapper, SoA) {
    constexpr std::size_t N = 32;
    int32_t x[N];
    int32_t y[N];
    int32_t z[N];
    wrapper::wrapper<Skeleton::Point3D, wrapper::pointer, wrapper::layout::soa> w = {{{{x, y}}, z}};
    test_random_access(N, w);
}

template <wrapper::layout L>
uint32_t compute_sum(int N, wrapper::wrapper<Skeleton::Point2D, std::span, L> w) {
    uint32_t sum = 0;
    for (int i = 0; i < N; ++i) {
        wrapper::wrapper<Skeleton::Point2D, wrapper::const_reference> cr = w[i];
        sum += cr.x + cr.y;
    }
    return sum;
}

TEST(OverloadResolution, AoS) {
    wrapper::wrapper<Skeleton::Point2D, std::vector, wrapper::layout::aos> w{
        {0, 5}, {1, 6}, {2, 7}, {3, 8}, {4, 9}
    };
    uint32_t sum = compute_sum<w.layout_type>(5, w);
    uint32_t expected_sum = 45;
    EXPECT_EQ(expected_sum, sum);
}
TEST(OverloadResolution, SoA) {
    wrapper::wrapper<Skeleton::Point2D, std::vector, wrapper::layout::soa> w{
        {{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}}
    };
    uint32_t sum = compute_sum<w.layout_type>(5, w);
    uint32_t expected_sum = 45;
    EXPECT_EQ(expected_sum, sum);
}

TEST(VectorToSpan, AoS) {
    wrapper::wrapper<Skeleton::Point2D, std::vector, wrapper::layout::aos> w{
        {0, 5}, {1, 6}, {2, 7}, {3, 8}, {4, 9}
    };
    wrapper::wrapper<Skeleton::Point2D, std::span, wrapper::layout::aos> w_span{w};
    uint32_t sum = compute_sum(5, w_span);
    uint32_t expected_sum = 45;
    EXPECT_EQ(expected_sum, sum);
}
TEST(VectorToSpan, SoA) {
    wrapper::wrapper<Skeleton::Point2D, std::vector, wrapper::layout::soa> w{
        {{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}}
    };
    wrapper::wrapper<Skeleton::Point2D, std::span, wrapper::layout::soa> w_span{w};
    uint32_t sum = compute_sum(5, w_span);
    uint32_t expected_sum = 45;
    EXPECT_EQ(expected_sum, sum);
}
