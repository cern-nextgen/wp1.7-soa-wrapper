#ifndef SKELETON_H
#define SKELETON_H

#include "wrapper.h"

namespace skeleton {

struct Point2D { double x, y; };

template <template <class> class F>
struct S {
    WRAPPER_APPLY_UNARY(x, y, point, identifier)
    WRAPPER_APPLY_BINARY(S, WRAPPER_EXPAND(x), WRAPPER_EXPAND(y), WRAPPER_EXPAND(point), WRAPPER_EXPAND(identifier))

    F<int> x;
    F<int> y;
    F<Point2D> point;
    F<double> identifier;  // std::string causes linker error

    constexpr int abs2() const { return x * x + y * y; }
    constexpr int& getX() { return x; }
    constexpr const int& getX() const { return x; }
    constexpr void setX(int x_new) { x = x_new; }
};

}

#endif  // SKELETON_H