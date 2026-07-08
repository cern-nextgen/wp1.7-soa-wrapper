#ifndef STRUCTS_H
#define STRUCTS_H

#include <algorithm>

#include "memlayout/wrapper.h"

namespace test {

template <template <class> class Container>
struct Point2D {
    MEMLAYOUT_APPLY(Point2D, x, y)

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
    MEMLAYOUT_APPLY(Point3D, point2d, z)

    memlayout::Wrapper<Point2D, Container> point2d;
    Container<int32_t> z;

    constexpr int32_t abs2() const { return point2d.abs2() + z * z; }
    constexpr int32_t& getZ() { return z; }
    constexpr const int32_t& getZ() const { return z; }
    constexpr void setZ(int32_t z_new) { z = z_new; }
};

template <template <class> class Container>
struct Gaussian {
    MEMLAYOUT_APPLY(Gaussian, mean, covariance)

    memlayout::Wrapper<Point3D, Container> mean;
    Container<uint16_t[3][3]> covariance;
};

}  // namespace test

#endif  // STRUCTS_H