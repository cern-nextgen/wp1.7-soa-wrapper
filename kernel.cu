#include "kernel.h"

#include <cmath>
#include <iostream>  // This include fixes segfault due to uninitialized std::cout, even if the latter is not used

#include "gpu.h"
#include "skeleton.h"
#include "wrapper.h"


namespace kernel {

void print_cuda_error(cudaError_t err) {
    if (err != cudaSuccess) std::cout << cudaGetErrorString(err) << std::endl;
}

template <class T>
struct raw_array {
    T *data;
    GPUd() T& operator[](std::size_t i) { return data[i]; }
    GPUd() const T& operator[](std::size_t i) const { return data[i]; }
};

__global__
void add(int N, wrapper::wrapper<raw_array, S, wrapper::layout::aos> w) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride) w[i].y = w[i].x + w[i].y;
}

int run() {
    int N = 8;
    wrapper::wrapper<raw_array, S, wrapper::layout::aos> w;
    print_cuda_error(cudaMallocManaged(&w.data.data, N * sizeof(S<wrapper::value>)));
    for (int i = 0; i < N; ++i) {
        S<wrapper::reference> r = w[i];
        r.setX(1);
        r.y = 2;
        r.point = {0.5 * i, 0.5 * i};
        r.identifier = 0.1 * i;
    }

    // int blockSize = 1;
    // int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<1, 1>>>(N, w);
    print_cuda_error(cudaDeviceSynchronize());

    int maxError = 0;
    for (int i = 0; i < N; ++i) maxError = std::max(maxError, std::abs(w[i].y - 3));
    print_cuda_error(cudaFree(w.data.data));
    return maxError;
}

}  // namespace kernel
