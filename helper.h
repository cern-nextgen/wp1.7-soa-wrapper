#ifndef HELPER_H
#define HELPER_H

#include <cstddef>

#include "gpu.h"


namespace helper {

namespace detail {

struct UniversalType {
    template<class T>
    GPUd() operator T() const;
};

template <typename T, typename Is, typename=void>
struct is_aggregate_constructible_from_n_impl : std::false_type {};

template <typename T, std::size_t...Is>
struct is_aggregate_constructible_from_n_impl<T, std::index_sequence<Is...>, std::void_t<decltype(T{(void(Is), UniversalType{})...})>> : std::true_type {};

template <typename T, std::size_t N>
using is_aggregate_constructible_from_n_helper = is_aggregate_constructible_from_n_impl<T, std::make_index_sequence<N>>;

template <typename T, std::size_t N>
struct is_aggregate_constructible_from_n {
    constexpr static bool value = is_aggregate_constructible_from_n_helper<T, N>::value && !is_aggregate_constructible_from_n_helper<T, N+1>::value;
};

}  // namespace detail

template <class T>
GPUd() constexpr std::size_t CountMembers() {
    if constexpr (detail::is_aggregate_constructible_from_n<T, 0>::value) return 0;
    else if (detail::is_aggregate_constructible_from_n<T,  1>::value) return  1;
    else if (detail::is_aggregate_constructible_from_n<T,  2>::value) return  2;
    else if (detail::is_aggregate_constructible_from_n<T,  3>::value) return  3;
    else if (detail::is_aggregate_constructible_from_n<T,  4>::value) return  4;
    else if (detail::is_aggregate_constructible_from_n<T,  5>::value) return  5;
    else if (detail::is_aggregate_constructible_from_n<T,  6>::value) return  6;
    else if (detail::is_aggregate_constructible_from_n<T,  7>::value) return  7;
    else if (detail::is_aggregate_constructible_from_n<T,  8>::value) return  8;
    else if (detail::is_aggregate_constructible_from_n<T,  9>::value) return  9;
    else if (detail::is_aggregate_constructible_from_n<T, 10>::value) return 10;
    else return 100;  // Silence warnings about missing return value
}

template <std::size_t M, class T, class S, class Functor>
GPUd() constexpr S apply_to_members(T t, Functor&& f) {
    if constexpr (M == 0) {
        return {};
    } else if constexpr (M == 1) {
        auto& [m00] = t;
        return {f(m00, 0)};
    } else if constexpr (M == 2) {
        auto& [m00, m01] = t;
        return {f(m00, 0), f(m01, 1)};
    } else if constexpr (M == 3) {
        auto& [m00, m01, m02] = t;
        return {f(m00, 0), f(m01, 1), f(m02, 2)};
    } else if constexpr (M == 4) {
        auto& [m00, m01, m02, m03] = t;
        return {f(m00, 0), f(m01, 1), f(m02, 2), f(m03, 3)};
    } else if constexpr (M == 5) {
        auto& [m00, m01, m02, m03, m04] = t;
        return {f(m00, 0), f(m01, 1), f(m02, 2), f(m03, 3), f(m04, 4)};
    } else if constexpr (M == 6) {
        auto& [m00, m01, m02, m03, m04, m05] = t;
        return {f(m00, 0), f(m01, 1), f(m02, 2), f(m03, 3), f(m04, 4), f(m05, 5)};
    } else if constexpr (M == 7) {
        auto& [m00, m01, m02, m03, m04, m05, m06] = t;
        return {f(m00, 0), f(m01, 1), f(m02, 2), f(m03, 3), f(m04, 4), f(m05, 5), f(m06, 6)};
    } else if constexpr (M == 8) {
        auto& [m00, m01, m02, m03, m04, m05, m06, m07] = t;
        return {f(m00, 0), f(m01, 1), f(m02, 2), f(m03, 3), f(m04, 4), f(m05, 5), f(m06, 6), f(m07, 7)};
    } else if constexpr (M == 9) {
        auto& [m00, m01, m02, m03, m04, m05, m06, m07, m08] = t;
        return {f(m00, 0), f(m01, 1), f(m02, 2), f(m03, 3), f(m04, 4), f(m05, 5), f(m06, 6), f(m07, 7), f(m08, 8)};
    } else if constexpr (M == 10) {
        auto& [m00, m01, m02, m03, m04, m05, m06, m07, m08, m09] = t;
        return {f(m00, 0), f(m01, 1), f(m02, 2), f(m03, 3), f(m04, 4), f(m05, 5), f(m06, 6), f(m07, 7), f(m08, 8), f(m09, 9)};
    }
}

template <std::size_t M, class T, class Functor>
GPUd() constexpr void apply_to_member_pairs(T left, T right, Functor&& f) {
    if constexpr (M == 0) {
        f();
    } else if constexpr (M == 1) {
        auto& [m00] = left;
        auto& [n00] = right;
        f(m00, n00);
    } else if constexpr (M == 2) {
        auto& [m00, m01] = left;
        auto& [n00, n01] = right;
        f(m00, n00); f(m01, n01);
    } else if constexpr (M == 3) {
        auto& [m00, m01, m02] = left;
        auto& [n00, n01, n02] = right;
        f(m00, n00); f(m01, n01); f(m02, n02);
    } else if constexpr (M == 4) {
        auto& [m00, m01, m02, m03] = left;
        auto& [n00, n01, n02, n03] = right;
        f(m00, n00); f(m01, n01); f(m02, n02); f(m03, n03);
    } else if constexpr (M == 5) {
        auto& [m00, m01, m02, m03, m04] = left;
        auto& [n00, n01, n02, n03, n04] = right;
        f(m00, n00); f(m01, n01); f(m02, n02); f(m03, n03); f(m04, n04);
    } else if constexpr (M == 6) {
        auto& [m00, m01, m02, m03, m04, m05] = left;
        auto& [n00, n01, n02, n03, n04, n05] = right;
        f(m00, n00); f(m01, n01); f(m02, n02); f(m03, n03); f(m04, n04); f(m05, n05);
    } else if constexpr (M == 7) {
        auto& [m00, m01, m02, m03, m04, m05, m06] = left;
        auto& [n00, n01, n02, n03, n04, n05, n06] = right;
        f(m00, n00); f(m01, n01); f(m02, n02); f(m03, n03); f(m04, n04); f(m05, n05); f(m06, n06);
    } else if constexpr (M == 8) {
        auto& [m00, m01, m02, m03, m04, m05, m06, m07] = left;
        auto& [n00, n01, n02, n03, n04, n05, n06, n07] = right;
        f(m00, n00); f(m01, n01); f(m02, n02); f(m03, n03); f(m04, n04); f(m05, n05); f(m06, n06); f(m07, n07);
    } else if constexpr (M == 9) {
        auto& [m00, m01, m02, m03, m04, m05, m06, m07, m08] = left;
        auto& [n00, n01, n02, n03, n04, n05, n06, n07, n08] = right;
        f(m00, n00); f(m01, n01); f(m02, n02); f(m03, n03); f(m04, n04); f(m05, n05); f(m06, n06); f(m07, n07); f(m08, n08);
    } else if constexpr (M == 10) {
        auto& [m00, m01, m02, m03, m04, m05, m06, m07, m08, m09] = left;
        auto& [n00, n01, n02, n03, n04, n05, n06, n07, n08, n09] = right;
        f(m00, n00); f(m01, n01); f(m02, n02); f(m03, n03); f(m04, n04); f(m05, n05); f(m06, n06); f(m07, n07); f(m08, n08); f(m09, n09);
    }
}

}  // namespace helper

#endif  // HELPER_H