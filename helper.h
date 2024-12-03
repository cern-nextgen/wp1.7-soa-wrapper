#ifndef HELPER_H
#define HELPER_H

#include <cstddef>

namespace helper {

namespace detail {

struct UniversalType {
    template<typename T>
    operator T() {}
};

}  // namespace detail

template<typename T>
consteval auto CountMembers(auto ...members) {
    if constexpr (requires { T{ members... }; } == false) return sizeof...(members) - 1;
    else return CountMembers<T>(members..., detail::UniversalType{});
}

template <std::size_t M, typename T, typename S, typename Functor>
constexpr S apply_to_members(T t, Functor&& f) {
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
    } /*else if constexpr (M == 5) {
        auto& [m00, m01, m02, m03, m04] = t;
        return {f(m00), f(m01), f(m02), f(m03), f(m04)};
    } else if constexpr (M == 6) {
        auto& [m00, m01, m02, m03, m04, m05] = t;
        return {f(m00), f(m01), f(m02), f(m03), f(m04), f(m05)};
    } else if constexpr (M == 7) {
        auto& [m00, m01, m02, m03, m04, m05, m06] = t;
        return {f(m00), f(m01), f(m02), f(m03), f(m04), f(m05), f(m06)};
    } else if constexpr (M == 8) {
        auto& [m00, m01, m02, m03, m04, m05, m06, m07] = t;
        return {f(m00), f(m01), f(m02), f(m03), f(m04), f(m05), f(m06), f(m07)};
    } else if constexpr (M == 9) {
        auto& [m00, m01, m02, m03, m04, m05, m06, m07, m08] = t;
        return {f(m00), f(m01), f(m02), f(m03), f(m04), f(m05), f(m06), f(m07), f(m08)};
    } else if constexpr (M == 10) {
        auto& [m00, m01, m02, m03, m04, m05, m06, m07, m08, m09] = t;
        return {f(m00), f(m01), f(m02), f(m03), f(m04), f(m05), f(m06), f(m07), f(m08), f(m09)};
    }*/
}

}  // namespace helper

#endif  // HELPER_H