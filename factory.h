#ifndef FACTORY_H
#define FACTORY_H

#include <array>
#include <numeric>
#include <memory_resource>
#include <type_traits>

#include "allocator.h"
#include "helper.h"
#include "kernel.h"
#include "wrapper.h"

namespace factory {

namespace pmr {

template <class T>
struct vector : std::vector<T, allocator::BufferAllocator<T>> {
    using Base = std::vector<T, allocator::BufferAllocator<T>>;
    vector(allocator::BufferAllocator<T> allocator) : Base(allocator.size_ / sizeof(T), allocator) {}
};

}

struct Buffer {
    template <class T>
    operator allocator::BufferAllocator<T>() { return {begin, size}; }
    char* begin;
    std::size_t size;
};

template <class T>
using AllBuffer = Buffer;

struct get_sizes {
    template <class... Args>
    std::array<std::size_t, sizeof...(Args)> operator()(Args&... args) const { return {sizeof(args)...}; }
};

template <
    template <template <class> class> class S,
    template <class> class F
>
struct from_aggregate {
    template <class... Args>
    S<F> operator()(Args... args) const { return {args...}; }
};

template <template <template <class> class> class S, wrapper::layout L>
constexpr std::size_t get_size_in_bytes() {
    if constexpr (L == wrapper::layout::aos) {
        return sizeof(S<wrapper::value>);
    } else if constexpr (L == wrapper::layout::soa) {
        S<wrapper::value> S_value;
        auto sizes = helper::invoke(S_value, get_sizes{});
        return std::reduce(sizes.cbegin(), sizes.cend());
    }
}

template <template <template <class> class> class S, wrapper::layout L>
wrapper::wrapper<S, pmr::vector, L> buffer_wrapper(char* buffer_ptr, std::size_t bytes) {
    if constexpr (L == wrapper::layout::aos) {
        return {allocator::BufferAllocator<S<wrapper::value>>(buffer_ptr, bytes)};
    } else if constexpr (L == wrapper::layout::soa) {
        constexpr static std::size_t M = helper::CountMembers<S<wrapper::value>>();

        S<wrapper::value> S_value;
        std::array<std::size_t, M> sizes = helper::invoke(S_value, get_sizes{});
        std::array<Buffer, M> buffers;
        std::size_t N = bytes / std::reduce(sizes.cbegin(), sizes.cend());

        std::size_t offset = 0;
        for (int m = 0; m < M; ++m) {
            std::size_t step = sizes[m] * N;
            buffers[m] = {buffer_ptr + offset, sizes[m]};
            offset += step;
        }

        S<AllBuffer> S_buffer = helper::invoke(buffers, from_aggregate<S, AllBuffer>{});

        return {S<allocator::BufferAllocator>(S_buffer)};
    }
}

struct call_constructor {
    std::size_t N;
    template <template <class> class F, class T>
    constexpr F<T> operator()(const F<T> &) const { return F<T>{N}; }
};

template <
    template <template <class> class> class S,
    template <class> class F,
    wrapper::layout L
>
wrapper::wrapper<S, F, L> default_wrapper(std::size_t N) {
    if constexpr (L == wrapper::layout::aos) return {{N}};
    else if constexpr (L == wrapper::layout::soa) {
        return { helper::invoke_on_members<F>(S<F>{}, call_constructor{N}) };
    }
}

}  // namespace factory

#endif  // FACTORY_H