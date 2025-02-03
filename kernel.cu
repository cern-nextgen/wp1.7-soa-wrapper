#include "kernel.h"

#include <iostream>
#include <math.h>
#include <vector>

#include "factory.h"
#include "gpu.h"
#include "skeleton.h"
#include "wrapper.h"


namespace kernel {

template <class T, std::size_t N>
struct raw_array {
    T *data;
    GPUd() T& operator[](std::size_t i) { return data[i]; }
    GPUd() const T& operator[](std::size_t i) const { return data[i]; }
};

template <template <class, std::size_t> class array_type, std::size_t N>
struct bind_size {
    template <class T>
    using type = array_type<T, N>;
};

constexpr int N = 256;

template <class T>
using my_array = bind_size<raw_array, N>::type<T>;

// Kernel function to add the elements of two arrays
__global__
void add(int n, wrapper::wrapper<my_array, S, wrapper::layout::aos> &w) {
    for (int i = 0; i < n; ++i) w[i].y = w[i].x + w[i].y;
}

float run() {
    my_array<S<wrapper::value>> buffer;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&buffer.data, N * sizeof(S<wrapper::value>));

    wrapper::wrapper<my_array, S, wrapper::layout::aos> w;

    // initialize on the host
    for (int i = 0; i < N; ++i) {
        S<wrapper::reference> r = w[i];
        r.setX(1.0f);
        r.y = 2.0;
        r.point = {0.5 * i, 0.5 * i};
        r.identifier = 0.1 * i;
    }

    // Run kernel on 1M elements on the GPU
    add<<<1, 1>>>(N, w);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) maxError = fmax(maxError, fabs(w[i].y - 3.0f));

    // Free memory
    cudaFree(buffer.data);

    return maxError;
}

}  // namespace kernel
