#include <span>

#include <gtest/gtest.h>

#include "debug.h"
#include "kernel.h"
#include "skeleton.h"
#include "wrapper.h"

template <wrapper::layout L>
__global__ void add(int N, wrapper::wrapper<Skeleton::Point3D, std::span, L> w) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride) w[i].z = w[i].point2d.x + w[i].point2d.y;
}

TEST(DeviceSpanWrapper, AoS) {
    constexpr std::size_t N = 32;

    wrapper::wrapper<Skeleton::Point3D, debug::vector, wrapper::layout::aos> h_w{N};
    for (int i = 0; i < N; ++i) {
        wrapper::wrapper<Skeleton::Point3D, wrapper::reference> r = h_w[i];
        h_w[i].point2d.x = i - 2;
        h_w[i].point2d.y = i + 2;
        h_w[i].z = 3 * i;
    }

    wrapper::wrapper<Skeleton::Point3D, kernel::unique_dev_ptr, wrapper::layout::aos> d_w{N};
    cudaMemcpy(d_w.get(), h_w.data(), N * sizeof(Skeleton::Point3D<wrapper::value>), cudaMemcpyHostToDevice);

    wrapper::wrapper<Skeleton::Point3D, std::span, wrapper::layout::aos> span_w(d_w);
    add<<<1, 1>>>(N, span_w);
    cudaError_t cuda_error = cudaDeviceSynchronize();
    EXPECT_EQ(cuda_error, 0);

    cudaMemcpy(h_w.data(), d_w.get(), N * sizeof(Skeleton::Point3D<wrapper::value>), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i) EXPECT_EQ(h_w[i].z, 2 * i);
}
TEST(DeviceSpanWrapper, SoA) {
    constexpr std::size_t N = 32;
    
    wrapper::wrapper<Skeleton::Point3D, debug::vector, wrapper::layout::soa> h_w{{{{N, N}}, N}};
    for (int i = 0; i < N; ++i) {
        wrapper::wrapper<Skeleton::Point3D, wrapper::reference> r = h_w[i];
        h_w[i].point2d.x = i - 2;
        h_w[i].point2d.y = i + 2;
        h_w[i].z = 3 * i;
    }

    wrapper::wrapper<Skeleton::Point3D, kernel::unique_dev_ptr, wrapper::layout::soa> d_w{{{{N, N}}, N}};
    cudaMemcpy(d_w.point2d.x.get(), h_w.point2d.x.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w.point2d.y.get(), h_w.point2d.y.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w.z.get(), h_w.z.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice);

    wrapper::wrapper<Skeleton::Point3D, std::span, wrapper::layout::soa> span_w = d_w;
    add<<<1, 1>>>(N, span_w);
    cudaError_t cuda_error = cudaDeviceSynchronize();
    EXPECT_EQ(cuda_error, 0);

    cudaMemcpy(h_w.point2d.x.data(), d_w.point2d.x.get(), N * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_w.point2d.y.data(), d_w.point2d.y.get(), N * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_w.z.data(), d_w.z.get(), N * sizeof(int32_t), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) EXPECT_EQ(h_w[i].z, 2 * i);
}