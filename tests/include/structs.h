#ifndef STRUCTS_H
#define STRUCTS_H

#include <algorithm>

#include "memlayout/wrapper.h"

namespace test {

template <template <class> class Container>
struct Point2D {
    MEMLAYOUT_APPLY_UNARY(x, y)
    MEMLAYOUT_APPLY_BINARY(Point2D, MEMLAYOUT_EXPAND(x), MEMLAYOUT_EXPAND(y))

    Container<int32_t> x, y;
    constexpr int32_t abs2() const { return x * x + y * y; }
};

void swap(Point2D<memlayout::reference> a, Point2D<memlayout::reference> b) {
    std::swap(a.x, b.x);
    std::swap(a.y, b.y);
}

struct Comp {
    using cref = const memlayout::Wrapper<Point2D, memlayout::const_reference>;
    constexpr bool operator()(cref& p, cref& q) const { return p.x < q.x; }
};

template <template <class> class Container>
struct Point3D {
    MEMLAYOUT_APPLY_UNARY(point2d, z)
    MEMLAYOUT_APPLY_BINARY(Point3D, MEMLAYOUT_EXPAND(point2d), MEMLAYOUT_EXPAND(z))

    memlayout::Wrapper<Point2D, Container> point2d;
    Container<int32_t> z;

    constexpr int32_t abs2() const { return point2d.abs2() + z * z; }
    constexpr int32_t& getZ() { return z; }
    constexpr const int32_t& getZ() const { return z; }
    constexpr void setZ(int32_t z_new) { z = z_new; }
};

template <template <class> class Container>
struct Gaussian {
    MEMLAYOUT_APPLY_UNARY(mean, covariance)
    MEMLAYOUT_APPLY_BINARY(Gaussian, MEMLAYOUT_EXPAND(mean), MEMLAYOUT_EXPAND(covariance))

    memlayout::Wrapper<Point3D, Container> mean;
    Container<uint16_t[3][3]> covariance;
};

}  // namespace test

#endif  // STRUCTS_H