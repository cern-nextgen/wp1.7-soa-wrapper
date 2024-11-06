#ifndef WRAPPER_H
#define WRAPPER_H

#include "aos.h"
#include "soa.h"

#include <concepts>


namespace wrapper {

template<template <std::size_t, class...> class F, template <template <std::size_t, class...> class> class S>
concept memberfunction_concept = std::same_as<S<F>, S<aos::identity>> || std::same_as<S<F>, S<soa::reference>>;

template<
    template <std::size_t, class...> class F,
    template <template <std::size_t, class...> class> class S,
    template <template <std::size_t, class...> class, template <template <std::size_t, class...> class> class> class Layout
>
using wrapper = Layout<F, S>;

}  // namespace wrapper

#endif  // WRAPPER_H