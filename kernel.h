#ifndef KERNEL_H
#define KERNEL_H

#include <memory>
#include <span>

#include "helper.h"
#include "wrapper.h"

namespace kernel {

int cuda_malloc_managed(void** data, std::size_t size);

int cuda_free(void* ptr);

int cuda_malloc(void** d_data, std::size_t size);

enum class cuda_memcpy_kind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost };
int cuda_memcpy(void* d_data, void* h_data, std::size_t size, cuda_memcpy_kind kind);

template <class T>
using span_type = std::span<T>;

template <template <template <class> class> class S>
void cuda_memcpy(
    wrapper::wrapper<span_type, S, wrapper::layout::aos> dst,
    wrapper::wrapper<span_type, S, wrapper::layout::aos> src,
    std::size_t N,
    cuda_memcpy_kind kind
) { cuda_memcpy(dst.data.data(), src.data.data(), N * sizeof(S<wrapper::value>), kind); }

template <template <template <class> class> class S>
void cuda_memcpy(
    wrapper::wrapper<span_type, S, wrapper::layout::soa> dst,
    wrapper::wrapper<span_type, S, wrapper::layout::soa> src,
    std::size_t N,
    cuda_memcpy_kind kind) {
    constexpr static std::size_t M = helper::CountMembers<S<wrapper::value>>();
    auto memcpy = [N, kind](auto dst_span, auto src_span) -> void {
        using value_type = decltype(src_span)::value_type;
        cuda_memcpy(dst_span.data(), src_span.data(), N * sizeof(value_type), kind);
    };
    using array_type = wrapper::wrapper<span_type, S, wrapper::layout::soa>::array_type;
    helper::apply_to_member_pairs<M, array_type>(dst.data, src.data, memcpy);
}

template <class T>
using pointer_type = T*;

template <
    template <class> class F,
    template <template <class> class> class S,
    wrapper::layout L
>
void apply(int N, wrapper::wrapper<F, S, L> w);

template<class T>
struct ManagedMemoryAllocator {
    using value_type = T;
    ManagedMemoryAllocator() = default;

    template <class U>
    ManagedMemoryAllocator(const ManagedMemoryAllocator<U>&) {}

    T* allocate(std::size_t n) {
        T* ptr;
        cuda_malloc_managed((void **) &ptr, sizeof(T) * n);
        return ptr;
    }

    void deallocate(T* p, std::size_t n) noexcept { cuda_free(p); }
};

template <class T>
struct device_memory_array {
    device_memory_array(std::size_t N) : ptr(nullptr, [](T * ptr){ kernel::cuda_free(ptr); }), N{N} {
        cuda_malloc((void **) &ptr, N * sizeof(T));
    }
    operator std::span<T>() { return { ptr.get(), ptr.get() + N }; }
    T operator[](std::size_t i) const { return *(ptr.get() + i); }
    T& operator[](std::size_t i) { return *(ptr.get() + i); }
    std::shared_ptr<T> ptr;
    std::size_t N;
};

}  // namespace kernel

#endif  // KERNEL_H