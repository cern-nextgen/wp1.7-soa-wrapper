#ifndef KERNEL_H
#define KERNEL_H

#include <span>

#include "skeleton.h"
#include "wrapper.h"

namespace kernel {

template <wrapper::layout L>
__global__ void add(int N, wrapper::wrapper<Skeleton::Point3D, std::span, L> w) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride) w[i].z = w[i].point2d.x + w[i].point2d.y;
}

template <class T>
struct unique_dev_ptr {
    unique_dev_ptr(int N) : ptr(), N{N} { cudaMalloc((void **) &ptr, N * sizeof(T)); }
    ~unique_dev_ptr() { if (ptr != nullptr) cudaFree(ptr); }
    unique_dev_ptr(const unique_dev_ptr& other) = delete;
    unique_dev_ptr& operator=(const unique_dev_ptr& other) = delete;
    unique_dev_ptr(unique_dev_ptr&& other) noexcept : ptr(other.ptr), N(other.N) {
        other.ptr = nullptr;
        other.N = 0;
    }
    unique_dev_ptr& operator=(unique_dev_ptr&& other) noexcept {
        ptr = other.ptr;
        N = other.N;
        other.ptr = nullptr;
        other.N = 0;
        return *this;
    }
    constexpr operator std::span<T>() { return { ptr, ptr + N }; }
    constexpr T& operator[](int i) { return ptr[i]; }
    constexpr const T& operator[](int i) const { return ptr[i]; }
    constexpr T*& get() { return ptr; }

  private:
    T* ptr;
    int N;
};

}  // namespace kernel

#endif  // KERNEL_H