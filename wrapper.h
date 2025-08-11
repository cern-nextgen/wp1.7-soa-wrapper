#ifndef WRAPPER_H
#define WRAPPER_H

#include "helper.h"

#include <cstddef>

namespace wrapper {

enum class layout { aos = 0, soa = 1 };

template <class T>
using value = T;

template <class T>
using reference = T&;

template <class T>
using const_reference = const T&;

template<
    template <template <class> class> class S,
    template <class> class F,
    layout L
>
struct wrapper;

template <template <template <class> class> class S, template <class> class F>
struct wrapper<S, F, layout::aos> {
    F<S<value>> data;

    template <template <class> class F_out>
    constexpr operator wrapper<S, F_out, layout::aos>() { return {data}; };

    constexpr S<reference> operator[](std::size_t i) { return data[i]; }
    constexpr S<const_reference> operator[](std::size_t i) const { return data[i]; }
};

template <template <template <class> class> class S, template <class> class F>
struct wrapper<S, F, layout::soa> : S<F> {
    template <template <class> class F_out>
    constexpr operator wrapper<S, F_out, layout::soa>() { return {*this}; };

    constexpr S<reference> operator[](std::size_t i) {
        return helper::invoke_on_members<reference, F>(*this, evaluate_at{i});
    }
    constexpr S<const_reference> operator[](std::size_t i) const {
        return helper::invoke_on_members<const_reference, F>(*this, evaluate_at{i});
    }

    private:

    struct evaluate_at {
        std::size_t i;

        template <template <class> class F_in, class T>
        constexpr reference<T> operator()(F_in<T> & t) const { return t[i]; }
        
        template <template <class> class F_in, class T>
        constexpr const_reference<T> operator()(const F_in<T> & t) const { return t[i]; }
    };
};

}  // namespace wrapper

#endif  // WRAPPER_H