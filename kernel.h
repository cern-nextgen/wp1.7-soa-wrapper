#ifndef KERNEL_H
#define KERNEL_H

#include <span>

#include "wrapper.h"

namespace kernel {

template <
    template <template <class> class> class S,
    wrapper::layout L
>
__global__ void add(int N, wrapper::wrapper<S, std::span, L> w) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride) w[i].y = w[i].getX() + w[i].y;
}

template <class T>
struct device_memory_array {
    device_memory_array(int N) : ptr(), N{N} { cudaMalloc((void **) &ptr, N * sizeof(T)); }
    ~device_memory_array() { if (ptr != nullptr) cudaFree(ptr); }
    constexpr operator std::span<T>() { return { ptr, ptr + N }; }
    constexpr T& operator[](int i) { return ptr[i]; }
    constexpr const T& operator[](int i) const { return ptr[i]; }
    T* ptr;
    int N;
};

}  // namespace kernel

#endif  // KERNEL_H