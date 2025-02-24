#ifndef KERNEL_H
#define KERNEL_H

#include <span>

#include "wrapper.h"

namespace kernel {

int cuda_malloc_managed(void** data, std::size_t size);

int cuda_free(void* ptr);

int cuda_malloc(void** d_data, int size);

enum class copy_flag { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost };
int cuda_memcpy(void* d_data, void* h_data, int size, copy_flag flag);

template <class T>
using span_type = std::span<T>;

template <class T>
using pointer_type = T*;

template <
    template <class> class F,
    template <template <class> class> class S,
    wrapper::layout L
>
void apply(int N, wrapper::wrapper<F, S, L> w);

template <template <template <class> class> class S, wrapper::layout L>
struct UnifiedMemoryManager;

template <template <template <class> class> class S>
struct UnifiedMemoryManager<S, wrapper::layout::aos> {
    UnifiedMemoryManager(std::size_t N) {
        cuda_malloc_managed((void **) &data, N * sizeof(S<wrapper::value>));
    }
    wrapper::wrapper<pointer_type, S, wrapper::layout::aos> create_wrapper() { return { data }; }
    ~UnifiedMemoryManager() { cuda_free(data); }
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
        GPUd() auto operator()(auto * member, std::size_t i) const {
            #ifndef __CUDA_ARCH__
            cuda_malloc_managed((void **) &member, N * sizeof(*member));
            #endif
            return member;
        }
        std::size_t N;
    };
    struct free_unified_memory {
        GPUd() auto operator()(auto * member, std::size_t i) const {
            #ifndef __CUDA_ARCH__
            cuda_free(member);
            #endif
            return nullptr;
        }
    };
    constexpr static std::size_t M = helper::CountMembers<S<wrapper::value>>();
    S<pointer_type> data;
};

}  // namespace kernel

#endif  // KERNEL_H