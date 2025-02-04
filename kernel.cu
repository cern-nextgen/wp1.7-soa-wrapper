#include "kernel.h"

#include <cmath>
#include <iostream>

#include "gpu.h"
#include "skeleton.h"
#include "wrapper.h"


namespace kernel {

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

float run() {
    int N = 8;
    wrapper::wrapper<raw_array, S, wrapper::layout::aos> w;

    cudaError_t err = cudaMallocManaged(&w.data.data, N * sizeof(S<wrapper::value>));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << cudaSuccess << ", " << err << std::endl;
    }


    // Initialize on the host
    for (int i = 0; i < N; ++i) {
        S<wrapper::reference> r = w[i];
        r.setX(1);
        r.y = 2;
        r.point = {0.5 * i, 0.5 * i};
        r.identifier = 0.1 * i;
    }

    // Run kernel
    // int blockSize = 1;
    // int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<1, 1>>>(N, w);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA cudaDeviceSynchronize failed: " << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << cudaSuccess << ", " << err << std::endl;
    }

    int maxError = 0;
    for (int i = 0; i < N; ++i) {
        //std::cout << "(" << w[i].x << ", " << w[i].y << "), ";
        maxError = std::max(maxError, std::abs(w[i].y - 3));
    }

    err = cudaFree(w.data.data);
    if (err != cudaSuccess) {
        std::cerr << "CUDA cudaFree failed: " << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << cudaSuccess << ", " << err << std::endl;
    }
    w.data.data = nullptr;

    return maxError;
}

}  // namespace kernel
