#include <span>

#include <gtest/gtest.h>

#include "debug.h"
#include "kernel.h"
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

TEST(DeviceSpanWrapper, AoS) {
    constexpr std::size_t N = 32;

    wrapper::wrapper<Skeleton::Point3D, debug::vector, wrapper::layout::aos> h_w{N};
    initialize(N, h_w);
    assert_equal_to_initialization(N, h_w);

    wrapper::wrapper<Skeleton::Point3D, kernel::unique_dev_ptr, wrapper::layout::aos> d_w{N};
    cudaMemcpy(d_w.get(), h_w.data(), N * sizeof(Skeleton::Point3D<wrapper::value>), cudaMemcpyHostToDevice);

    wrapper::wrapper<Skeleton::Point3D, std::span, wrapper::layout::aos> span_w(d_w);
    kernel::add<<<1, 1>>>(N, span_w);
    cudaError_t cuda_error = cudaDeviceSynchronize();
    EXPECT_EQ(cuda_error, 0);

    cudaMemcpy(h_w.data(), d_w.get(), N * sizeof(Skeleton::Point3D<wrapper::value>), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i) EXPECT_EQ(h_w[i].z, 2 * i);
}
TEST(DeviceSpanWrapper, SoA) {
    constexpr std::size_t N = 32;
    
    wrapper::wrapper<Skeleton::Point3D, debug::vector, wrapper::layout::soa> h_w{{{{N, N}}, N}};
    initialize(N, h_w);
    assert_equal_to_initialization(N, h_w);

    wrapper::wrapper<Skeleton::Point3D, kernel::unique_dev_ptr, wrapper::layout::soa> d_w{{{{N, N}}, N}};
    cudaMemcpy(d_w.point2d.x.get(), h_w.point2d.x.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w.point2d.y.get(), h_w.point2d.y.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w.z.get(), h_w.z.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice);

    wrapper::wrapper<Skeleton::Point3D, std::span, wrapper::layout::soa> span_w = d_w;
    kernel::add<<<1, 1>>>(N, span_w);
    cudaError_t cuda_error = cudaDeviceSynchronize();
    EXPECT_EQ(cuda_error, 0);

    cudaMemcpy(h_w.point2d.x.data(), d_w.point2d.x.get(), N * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_w.point2d.y.data(), d_w.point2d.y.get(), N * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_w.z.data(), d_w.z.get(), N * sizeof(int32_t), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) EXPECT_EQ(h_w[i].z, 2 * i);
}