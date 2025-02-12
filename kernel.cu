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
using pointer_type = T*;

template <
    template <class> class F,
    template <template <class> class> class S,
    wrapper::layout L
>
__global__ void add(int N, wrapper::wrapper<F, S, L> w) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride) w[i].y = w[i].getX() + w[i].y;
}

template <wrapper::layout L> struct UnifiedMemoryManager;

template <>
struct UnifiedMemoryManager<wrapper::layout::aos> {
    UnifiedMemoryManager(std::size_t N) { print_cuda_error(cudaMallocManaged(&data, N * sizeof(S<wrapper::value>))); }
    wrapper::wrapper<pointer_type, S, wrapper::layout::aos> create_wrapper() { return { data }; }
    ~UnifiedMemoryManager() { print_cuda_error(cudaFree(data)); }
    private:
    pointer_type<S<wrapper::value>> data;
};

template <>
struct UnifiedMemoryManager<wrapper::layout::soa> {
    UnifiedMemoryManager(std::size_t N) {
        print_cuda_error(cudaMallocManaged(&x, N * sizeof(int)));
        print_cuda_error(cudaMallocManaged(&y, N * sizeof(int)));
        print_cuda_error(cudaMallocManaged(&point, N * sizeof(Point2D)));
        print_cuda_error(cudaMallocManaged(&identifier, N * sizeof(double)));
    }
    wrapper::wrapper<pointer_type, S, wrapper::layout::soa> create_wrapper() { return { x, y, point, identifier }; }
    ~UnifiedMemoryManager() {
        print_cuda_error(cudaFree(x));
        print_cuda_error(cudaFree(y));
        print_cuda_error(cudaFree(point));
        print_cuda_error(cudaFree(identifier));
    }
    private:
    pointer_type<int> x;
    pointer_type<int> y;
    pointer_type<Point2D> point;
    pointer_type<double> identifier;
};

template <wrapper::layout L>
int unified_memory_test() {
    std::size_t N = 8;
    UnifiedMemoryManager<L> unified_memory_manager(N);
    auto w = unified_memory_manager.create_wrapper();

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

    return maxError;
}

int run() {
    int error_aos = unified_memory_test<wrapper::layout::aos>();
    int error_soa = unified_memory_test<wrapper::layout::soa>();
    return std::max(error_aos, error_soa);
}

}  // namespace kernel
