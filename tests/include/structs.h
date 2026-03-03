#ifndef STRUCTS_H
#define STRUCTS_H

#include "memlayout/wrapper.h"

namespace test {

template <template <class> class Container>
struct Point2D {
    WRAPPER_APPLY_UNARY(x, y)
    WRAPPER_APPLY_BINARY(Point2D, WRAPPER_EXPAND(x), WRAPPER_EXPAND(y))

    Container<int32_t> x, y;
    constexpr int32_t abs2() const { return x * x + y * y; }
};

template <template <class> class Container>
struct Point3D {
    WRAPPER_APPLY_UNARY(point2d, z)
    WRAPPER_APPLY_BINARY(Point3D, WRAPPER_EXPAND(point2d), WRAPPER_EXPAND(z))

    memlayout::Wrapper<Point2D, Container> point2d;
    Container<int32_t> z;

    constexpr int32_t abs2() const { return point2d.abs2() + z * z; }
    constexpr int32_t& getZ() { return z; }
    constexpr const int32_t& getZ() const { return z; }
    constexpr void setZ(int32_t z_new) { z = z_new; }
};

template <template <class> class Container>
struct Gaussian {
    WRAPPER_APPLY_UNARY(mean, covariance)
    WRAPPER_APPLY_BINARY(Gaussian, WRAPPER_EXPAND(mean), WRAPPER_EXPAND(covariance))

    memlayout::Wrapper<Point3D, Container> mean;
    Container<uint16_t[3][3]> covariance;
};

}  // namespace test

#endif  // STRUCTS_H