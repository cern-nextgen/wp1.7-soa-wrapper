#ifndef SKELETON_H
#define SKELETON_H

#include "wrapper.h"

namespace Skeleton {

template <template <class> class F>
struct Point2D {
    WRAPPER_APPLY_UNARY(x, y)
    WRAPPER_APPLY_BINARY(Point2D, WRAPPER_EXPAND(x), WRAPPER_EXPAND(y))

    F<int32_t> x, y;
    constexpr int32_t abs2() const { return x * x + y * y; }
};

template <template <class> class F>
struct Point3D {
    WRAPPER_APPLY_UNARY(point2d, z)
    WRAPPER_APPLY_BINARY(Point3D, WRAPPER_EXPAND(point2d), WRAPPER_EXPAND(z))

    wrapper::wrapper<Point2D, F> point2d;
    F<int32_t> z;

    constexpr int32_t abs2() const { return point2d.abs2() + z * z; }
    constexpr int32_t& getZ() { return z; }
    constexpr const int32_t& getZ() const { return z; }
    constexpr void setZ(int32_t z_new) { z = z_new; }
};

template <template <class> class F>
struct Position {
    WRAPPER_APPLY_UNARY(mean, covariance)
    WRAPPER_APPLY_BINARY(Position, WRAPPER_EXPAND(mean), WRAPPER_EXPAND(covariance))

    wrapper::wrapper<Point3D, F> mean;
    F<uint16_t[3][3]> covariance;
};

}

#endif  // SKELETON_H