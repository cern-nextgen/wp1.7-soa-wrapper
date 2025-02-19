#include "kernel.h"

#include <cmath>
#include <iostream>  // This include fixes segfault due to uninitialized std::cout, even if the latter is not used
#include <memory>

#include "gpu.h"
#include "skeleton.h"
#include "wrapper.h"

#include <cuda/std/span>
#include <vector>


namespace kernel {

template <class T>
using pointer_type = T*;

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

template <template <template <class> class> class S, wrapper::layout L>
struct UnifiedMemoryManager;

template <template <template <class> class> class S>
struct UnifiedMemoryManager<S, wrapper::layout::aos> {
    UnifiedMemoryManager(std::size_t N) { print_cuda_error(cudaMallocManaged(&data, N * sizeof(S<wrapper::value>))); }
    wrapper::wrapper<pointer_type, S, wrapper::layout::aos> create_wrapper() { return { data }; }
    ~UnifiedMemoryManager() { print_cuda_error(cudaFree(data)); }
    private:
    pointer_type<S<wrapper::value>> data;
};

template <template <template <class> class> class S>
struct UnifiedMemoryManager<S, wrapper::layout::soa> {
    UnifiedMemoryManager(std::size_t N) {
        data = helper::apply_to_members<M, S<pointer_type>, S<pointer_type>>(data, allocate_unified_memory(N));
    }
    wrapper::wrapper<pointer_type, S, wrapper::layout::soa> create_wrapper() { return { data }; }
    ~UnifiedMemoryManager() {
        data = helper::apply_to_members<M, S<pointer_type>, S<pointer_type>>(data, free_unified_memory{});
    }
    private:
    struct allocate_unified_memory {
        allocate_unified_memory(std::size_t N) : N(N) {};
        GPUd() auto operator()(auto member, std::size_t i) const {
            #ifndef __CUDA_ARCH__
            print_cuda_error(cudaMallocManaged(&member, N * sizeof(*member)));
            #endif
            return member;
        }
        std::size_t N;
    };
    struct free_unified_memory {
        GPUd() auto operator()(auto member, std::size_t i) const {
            #ifndef __CUDA_ARCH__
            print_cuda_error(cudaFree(member));
            #endif
            return nullptr;
        }
    };
    constexpr static std::size_t M = helper::CountMembers<S<wrapper::value>>();
    S<pointer_type> data;
};

template <
    template <class> class F,
    template <template <class> class> class S,
    wrapper::layout L
>
void apply_add(int N, wrapper::wrapper<F, S, L> w) {
    add<<<1, 1>>>(N, w);
    print_cuda_error(cudaDeviceSynchronize());
}

template <wrapper::layout L>  // void (*T)(), 
int unified_memory_test() {
    std::size_t N = 8;
    UnifiedMemoryManager<S, L> unified_memory_manager(N);
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
    apply_add(N, w);

    int maxError = 0;
    for (int i = 0; i < N; ++i) maxError = std::max(maxError, std::abs(w[i].y - 3));

    return maxError;
}

template <class T>
void deleter(T* p) { cudaFree(p); };

template <class T>
using unified_memory_unique_ptr = std::unique_ptr<T[], decltype(&deleter<T>)>;

template <class T>
struct UnifiedMemoryArray {
    UnifiedMemoryArray(std::size_t N) { print_cuda_error(cudaMallocManaged(&ptr, N * sizeof(T))); }
    ~UnifiedMemoryArray() { } //cudaFree(ptr); }
    GPUd() T& operator[](std::size_t i) { return ptr[i]; }
    GPUd() const T& operator[](std::size_t i) const { return ptr[i]; }
    //private:
    T* ptr;
};

int run() {
    int error_aos = unified_memory_test<wrapper::layout::aos>();
    int error_soa = unified_memory_test<wrapper::layout::soa>();

    /*S<wrapper::value> * data;
    std::size_t N = 8;
    std::size_t size = sizeof(S<wrapper::value>);
    print_cuda_error(cudaMallocManaged(&data, N * size));
    wrapper::wrapper<cuda::std::span, S, wrapper::layout::aos> w{{data, data + N * size}};

    for (int i = 0; i < N; ++i) {
        S<wrapper::reference> r = w[i];
        r.setX(1);
        r.y = 2;
        r.point = {0.5 * i, 0.5 * i};
        r.identifier = 0.1 * i;
    }

    add<<<1, 1>>>(N, w);
    print_cuda_error(cudaDeviceSynchronize());

    int maxError = 0;
    for (int i = 0; i < N; ++i) maxError = std::max(maxError, std::abs(w[i].y - 3));
    cudaFree(data);

    return maxError; */
    return std::max(error_aos, error_soa);
}

}  // namespace kernel
