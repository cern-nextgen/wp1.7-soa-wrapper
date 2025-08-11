#ifndef HELPER_H
#define HELPER_H

#include <cstddef>

namespace helper {

namespace detail {

struct UniversalType {
    template<class T>
    operator T() const;
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

template <typename T>
constexpr bool false_type = false;

}  // namespace detail

template <class Argument>
constexpr std::size_t CountMembers() {
    if constexpr (detail::is_aggregate_constructible_from_n<Argument, 0>::value) return 0;
    else if constexpr (detail::is_aggregate_constructible_from_n<Argument,  1>::value) return  1;
    else if constexpr (detail::is_aggregate_constructible_from_n<Argument,  2>::value) return  2;
    else if constexpr (detail::is_aggregate_constructible_from_n<Argument,  3>::value) return  3;
    else if constexpr (detail::is_aggregate_constructible_from_n<Argument,  4>::value) return  4;
    else if constexpr (detail::is_aggregate_constructible_from_n<Argument,  5>::value) return  5;
    else if constexpr (detail::is_aggregate_constructible_from_n<Argument,  6>::value) return  6;
    else if constexpr (detail::is_aggregate_constructible_from_n<Argument,  7>::value) return  7;
    else if constexpr (detail::is_aggregate_constructible_from_n<Argument,  8>::value) return  8;
    else if constexpr (detail::is_aggregate_constructible_from_n<Argument,  9>::value) return  9;
    else if constexpr (detail::is_aggregate_constructible_from_n<Argument, 10>::value) return 10;
    else {
        static_assert(detail::false_type<Argument>, "Unsupported number of members.");
        return 100;  // Silence warnings about missing return value
    }
}

template <
    class Argument,
    class FunctionObject
>
constexpr auto invoke(Argument & arg, FunctionObject&& f) {
    constexpr std::size_t M = helper::CountMembers<Argument>();
    if constexpr (M == 0) {
        return f();
    } else if constexpr (M == 1) {
        auto& [m00] = arg;
        return f(m00);
    } else if constexpr (M == 2) {
        auto& [m00, m01] = arg;
        return f(m00, m01);
    } else if constexpr (M == 3) {
        auto& [m00, m01, m02] = arg;
        return f(m00, m01, m02);
    } else if constexpr (M == 4) {
        auto& [m00, m01, m02, m03] = arg;
        return f(m00, m01, m02, m03);
    } else if constexpr (M == 5) {
        auto& [m00, m01, m02, m03, m04] = arg;
        return f(m00, m01, m02, m03, m04);
    } else if constexpr (M == 6) {
        auto& [m00, m01, m02, m03, m04, m05] = arg;
        return f(m00, m01, m02, m03, m04, m05);
    } else if constexpr (M == 7) {
        auto& [m00, m01, m02, m03, m04, m05, m06] = arg;
        return f(m00, m01, m02, m03, m04, m05, m06);
    } else if constexpr (M == 8) {
        auto& [m00, m01, m02, m03, m04, m05, m06, m07] = arg;
        return f(m00, m01, m02, m03, m04, m05, m06, m07);
    } else if constexpr (M == 9) {
        auto& [m00, m01, m02, m03, m04, m05, m06, m07, m08] = arg;
        return f(m00, m01, m02, m03, m04, m05, m06, m07, m08);
    } else if constexpr (M == 10) {
        auto& [m00, m01, m02, m03, m04, m05, m06, m07, m08, m09] = arg;
        return f(m00, m01, m02, m03, m04, m05, m06, m07, m08, m09);
    } else {
        static_assert(detail::false_type<Argument>, "Unsupported number of members.");
        return void();  // Silence warnings about missing return value
    }
}

template <
    template <class> class F_out,
    template <class> class F_in,
    template <template <class> class> class S,
    class  FunctionObject
>
struct memberwise {
    FunctionObject f;

    template <class... Args>  // HACK: NVCC cannot deduce template parameters of f.operator() like so: { f(args)... }
    constexpr S<F_out> operator()(Args&... args) const { return {f.template operator()<F_in>(args)...}; }
};

template <
    template <class> class F_out,
    template <class> class F_in,
    template <template <class> class> class S,
    class  FunctionObject
>
constexpr S<F_out> invoke_on_members(S<F_in> & s, FunctionObject&& f) {
    return invoke(s, memberwise<F_out, F_in, S, FunctionObject>{f});
}

template <
    template <class> class F_out,
    template <class> class F_in,
    template <template <class> class> class S,
    class  FunctionObject
>
constexpr S<F_out> invoke_on_members(const S<F_in> & s, FunctionObject&& f) {
    return invoke(s, memberwise<F_out, F_in, S, FunctionObject>{f});
}

}  // namespace helper

#endif  // HELPER_H