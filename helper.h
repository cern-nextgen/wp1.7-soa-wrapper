#include <cstddef>
#include <type_traits>
#include <utility>


#ifndef HELPER_H
#define HELPER_H


namespace helper {

namespace detail {

struct Anything {
    template <typename T>
    operator T() const;
};

template <typename T, typename Is, typename=void>
struct is_aggregate_constructible_from_n_impl : std::false_type {};

template <typename T, std::size_t...Is>
struct is_aggregate_constructible_from_n_impl<T, std::index_sequence<Is...>, std::void_t<decltype(T{(void(Is), Anything{})...})>> : std::true_type {};

template <typename T, std::size_t N>
using is_aggregate_constructible_from_n = is_aggregate_constructible_from_n_impl<T, std::make_index_sequence<N>>;

template <typename T, typename S, std::size_t N>
using enable_if_helper = typename std::enable_if<is_aggregate_constructible_from_n<T, N>::value && !is_aggregate_constructible_from_n<T, N+1>::value, S>;

}  // namespace detail

template <typename T, typename S, typename Functor>
constexpr typename detail::enable_if_helper<T, S, 0>::type apply_to_members(T& t, Functor&& f) { return S{}; }

template <typename T, typename S, typename Functor>
constexpr typename detail::enable_if_helper<T, S, 1>::type apply_to_members(T& t, Functor&& f) {
    auto& [m00] = t;
    return {f(m00)};
}

template <typename T, typename S, typename Functor>
constexpr typename detail::enable_if_helper<T, S, 2>::type apply_to_members(T& t, Functor&& f) {
    auto& [m00, m01] = t;
    return {f(m00), f(m01)};
}

template <typename T, typename S, typename Functor>
constexpr typename detail::enable_if_helper<T, S, 3>::type apply_to_members(T& t, Functor&& f) {
    auto& [m00, m01, m02] = t;
    return {f(m00), f(m01), f(m02)};
}

template <typename T, typename S, typename Functor>
constexpr typename detail::enable_if_helper<T, S, 4>::type apply_to_members(T& t, Functor&& f) {
    auto& [m00, m01, m02, m03] = t;
    return {f(m00), f(m01), f(m02), f(m03)};
}

template <typename T, typename S, typename Functor>
constexpr typename detail::enable_if_helper<T, S, 5>::type apply_to_members(T& t, Functor&& f) {
    auto& [m00, m01, m02, m03, m04] = t;
    return {f(m00), f(m01), f(m02), f(m03), f(m04)};
}

template <typename T, typename S, typename Functor>
constexpr typename detail::enable_if_helper<T, S, 6>::type apply_to_members(T& t, Functor&& f) {
    auto& [m00, m01, m02, m03, m04, m05] = t;
    return {f(m00), f(m01), f(m02), f(m03), f(m04), f(m05)};
}

template <typename T, typename S, typename Functor>
constexpr typename detail::enable_if_helper<T, S, 7>::type apply_to_members(T& t, Functor&& f) {
    auto& [m00, m01, m02, m03, m04, m05, m06] = t;
    return {f(m00), f(m01), f(m02), f(m03), f(m04), f(m05), f(m06)};
}

template <typename T, typename S, typename Functor>
constexpr typename detail::enable_if_helper<T, S, 8>::type apply_to_members(T& t, Functor&& f) {
    auto& [m00, m01, m02, m03, m04, m05, m06, m07] = t;
    return {f(m00), f(m01), f(m02), f(m03), f(m04), f(m05), f(m06), f(m07)};
}

template <typename T, typename S, typename Functor>
constexpr typename detail::enable_if_helper<T, S, 9>::type apply_to_members(T& t, Functor&& f) {
    auto& [m00, m01, m02, m03, m04, m05, m06, m07, m08] = t;
    return {f(m00), f(m01), f(m02), f(m03), f(m04), f(m05), f(m06), f(m07), f(m08)};
}

template <typename T, typename S, typename Functor>
constexpr typename detail::enable_if_helper<T, S, 10>::type apply_to_members(T& t, Functor&& f) {
    auto& [m00, m01, m02, m03, m04, m05, m06, m07, m08, m09] = t;
    return {f(m00), f(m01), f(m02), f(m03), f(m04), f(m05), f(m06), f(m07), f(m08), f(m09)};
}

}  // namespace helper

#endif  // HELPER_H