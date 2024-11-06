#ifndef WRAPPER_H
#define WRAPPER_H

#include "aos.h"
#include "soa.h"

#include <concepts>


namespace wrapper {

template<
    template <class> class F,
    template <template <class> class> class S,
    template <template <class> class, template <template <class> class> class> class Layout
>
using wrapper = Layout<F, S>;

template<template <class> class F, template <template <class> class> class S>
using aos = aos::aos<F, S>;

template<template <class> class F, template <template <class> class> class S>
using soa = soa::soa<F, S>;

}  // namespace wrapper

#endif  // WRAPPER_H