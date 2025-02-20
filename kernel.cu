#include "kernel.h"

#include <iostream>  // This include fixes segfault due to uninitialized std::cout, even if the latter is not used

#include "gpu.h"
#include "skeleton.h"  // Needed only for forward declarations
#include "wrapper.h"

// #include <cuda/std/span>  // Should work out of the box


namespace kernel {

template <class T>
using pointer_type = T*;

int cuda_malloc_managed(void** data, std::size_t size) { return cudaMallocManaged(data, size); }

int cuda_free(void* ptr) { return cudaFree(ptr); }

void print_cuda_error(cudaError_t err) {
    if (err != cudaSuccess) std::cout << cudaGetErrorString(err) << std::endl;
}

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

template <
    template <class> class F,
    template <template <class> class> class S,
    wrapper::layout L
>
void apply(int N, wrapper::wrapper<F, S, L> w) {
    add<<<1, 1>>>(N, w);
    print_cuda_error(cudaDeviceSynchronize());
}

// Explicit instatiations needed for unit tests (TODO: Get rid of this)
template void apply<pointer_type, S, wrapper::layout::aos>(int N, wrapper::wrapper<pointer_type, S, wrapper::layout::aos> w);
template void apply<pointer_type, S, wrapper::layout::soa>(int N, wrapper::wrapper<pointer_type, S, wrapper::layout::soa> w);

}  // namespace kernel
