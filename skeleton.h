#ifndef SKELETON_H
#define SKELETON_H

#include "gpu.h"

struct Point2D { double x, y; };

template <template <class> class F>
struct S {
    template<template <class> class F_new>
    constexpr operator S<F_new>() { return {x, y, point, identifier}; }
    template<template <class> class F_new>
    constexpr operator const S<F_new>() const { return {x, y, point, identifier}; }
    F<int> x;
    F<int> y;
    F<Point2D> point;
    F<double> identifier;  // std::string causes linker error

    GPUd() int abs2() const { return x * x + y * y; }
    GPUd() int& getX() { return x; }
    GPUd() const int& getX() const { return x; }
    GPUd() void setX(int x_new) { x = x_new; }
};

#endif  // SKELETON_H