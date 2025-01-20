#ifndef FACTORY_H
#define FACTORY_H

#include "allocator.h"
#include "helper.h"
#include "wrapper.h"

#include <type_traits>
#include <numeric>
#include <array>
#include <memory_resource>

namespace factory {

namespace pmr {

template <class T>
class vector {
    using data_type = std::pmr::vector<T>;
    std::unique_ptr<std::pmr::memory_resource> resource_m;
    std::pmr::vector<T> data_m;
public:
    vector(std::size_t size, std::unique_ptr<std::pmr::memory_resource> resource)
            : resource_m(std::move(resource)), data_m(size, resource_m.get()) {}
    data_type::reference operator[](std::size_t i) { return data_m[i]; }
    data_type::const_reference operator[](std::size_t i) const { return data_m[i]; }
};

}

template <template <template <class> class> class S>
wrapper::wrapper<pmr::vector, S, wrapper::layout::aos> make_wrapper_aos(char* buffer, std::size_t bytes) {
    std::size_t size = bytes / sizeof(S<wrapper::value>);
    auto resource_ptr = std::make_unique<allocator::BufferResource>(buffer, bytes);
    pmr::vector<S<wrapper::value>> data(size, std::move(resource_ptr));
    return wrapper::wrapper<pmr::vector, S, wrapper::layout::aos>(std::move(data));
}

template <template <template <class> class> class S>
wrapper::wrapper<pmr::vector, S, wrapper::layout::soa> make_wrapper_soa(char* buffer, std::size_t bytes) {
    constexpr std::size_t M = helper::CountMembers<S<wrapper::value>>();
    auto size_of = [](auto& member, std::size_t) -> std::size_t { return sizeof(member); };
    std::array<std::size_t, M> member_bytes = helper::apply_to_members<M, S<wrapper::value>, std::array<std::size_t, M>>(S<wrapper::value>(), size_of);
    std::size_t N = bytes / std::reduce(member_bytes.cbegin(), member_bytes.cend());

    std::array<std::size_t, M + 1> buffer_bytes = {0};
    for (int i = 1; i < M + 1; ++i) buffer_bytes[i] = member_bytes[i - 1] * N + buffer_bytes[i - 1];

    auto test = [N, buffer, buffer_bytes](auto& member, std::size_t m) -> decltype(auto) {
        auto resource_ptr = std::make_unique<allocator::BufferResource>(buffer + buffer_bytes[m], buffer_bytes[m + 1] - buffer_bytes[m]);
        return pmr::vector<typename std::remove_reference<decltype(member)>::type>(N, std::move(resource_ptr));
    };

    return helper::apply_to_members<M, S<wrapper::value>, wrapper::wrapper<pmr::vector, S, wrapper::layout::soa>>(S<wrapper::value>(), test);
}

template <template <template <class> class> class S, wrapper::layout L>
wrapper::wrapper<pmr::vector, S, L> make_wrapper(char* buffer, std::size_t bytes) {
    if constexpr (L == wrapper::layout::aos) return make_wrapper_aos<S>(buffer, bytes);
    else if constexpr (L == wrapper::layout::soa) return make_wrapper_soa<S>(buffer, bytes);
}

template <
    template <class> class F,
    template <template <class> class> class S
>
wrapper::wrapper<F, S, wrapper::layout::aos> default_aos_wrapper(std::size_t N) {
    return { F<S<wrapper::value>>(N) };
}

template <
    template <class> class F,
    template <template <class> class> class S
>
wrapper::wrapper<F, S, wrapper::layout::soa> default_soa_wrapper(std::size_t N) {
    constexpr static std::size_t M = helper::CountMembers<S<wrapper::value>>();
    auto forward_to_F_constructor = [N](auto member, std::size_t) -> decltype(auto) { return F<decltype(member)>(N); };
    return { helper::apply_to_members<M, S<wrapper::value>, S<F>>(S<wrapper::value>(), forward_to_F_constructor)  };
}

template <
    template <class> class F,
    template <template <class> class> class S,
    wrapper::layout L
>
wrapper::wrapper<F, S, L> default_wrapper(std::size_t N) {
    if constexpr (L == wrapper::layout::aos) return default_aos_wrapper<F, S>(N);
    else if constexpr (L == wrapper::layout::soa) return default_soa_wrapper<F, S>(N);
}

}  // namespace factory

#endif  // FACTORY_H