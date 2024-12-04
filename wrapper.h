#ifndef WRAPPER_H
#define WRAPPER_H

#include "helper.h"

#include <cstddef>

namespace wrapper {

enum class layout { aos = 0, soa = 1 };

template <class T>
using value = T;

template <class T>
using reference = T&;  // std::vector<T>::reference;

template <class T>
using const_reference = const T&;  // std::vector<T>::const_reference;

template <template <class> class F_in, template <template <class> class> class S>
struct proxy_type : S<F_in> {
    constexpr static std::size_t M = helper::CountMembers<S<value>>();
    template<template <class> class F_out>
    operator S<F_out>() const {
        auto id = [](auto& member, std::size_t) -> decltype(auto) { return member; };
        return helper::apply_to_members<M, S<F_in>, S<F_out>>(*this, id);
    }
};

template<
    template <class> class F,
    template <template <class> class> class S,
    layout L
>
struct wrapper;

template <template <class> class F, template <template <class> class> class S>
struct wrapper<F, S, layout::aos> {
    using value_type = S<value>;
    using array_type = F<value_type>;

    constexpr static std::size_t M = helper::CountMembers<value_type>();

    array_type data;

    value_type& get_reference(std::size_t i) { return data[i]; }
    const value_type& get_reference(std::size_t i) const { return data[i]; }

    proxy_type<reference, S> operator[](std::size_t i) {
        auto id = [](auto& member, std::size_t) -> decltype(auto) { return member; };
        return helper::apply_to_members<M, value_type&, proxy_type<reference, S>>(data[i], id);
    }
    proxy_type<const_reference, S> operator[](std::size_t i) const {
        auto id = [](const auto& member, std::size_t) -> decltype(auto) { return member; };
        return helper::apply_to_members<M, const value_type&, proxy_type<const_reference, S>>(data[i], id);
    }
};

template <template <class> class F, template <template <class> class> class S>
struct wrapper<F, S, layout::soa> {
    using value_type = S<value>;
    using array_type = S<F>;

    constexpr static std::size_t M = helper::CountMembers<value_type>();

    array_type data;

    proxy_type<reference, S> operator[](std::size_t i) {
        auto evaluate_at = [i](auto& member, std::size_t) -> decltype(auto) { return member[i]; };
        return helper::apply_to_members<M, array_type&, proxy_type<reference, S>>(data, evaluate_at);
    }
    proxy_type<const_reference, S> operator[](std::size_t i) const {
        auto evaluate_at = [i](const auto& member, std::size_t) -> decltype(auto) { return member[i]; };
        return helper::apply_to_members<M, const array_type&, proxy_type<const_reference, S>>(data, evaluate_at);
    }
};

}  // namespace wrapper

#endif  // WRAPPER_H