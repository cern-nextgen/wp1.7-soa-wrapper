#include "allocator.h"
#include "debug.h"
#include "factory.h"
#include "wrapper.h"

#include <iostream>
#include <span>
#include <string>

#include <gtest/gtest.h>


struct Point2D { double x, y; };

template <template <class> class F>
struct S {
    F<int> x;
    F<int> y;
    F<Point2D> point;
    F<std::string> identifier;

    int abs2() const { return x * x + y * y; }
    int& getX() { return x; }
    const int& getX() const { return x; }
    void setX(auto x_new) { x = x_new; }
};

template <class L, class R>
bool operator==(L l, R r) { return l.x == r.x && l.y == r.y && l.point == r.point && l.identifier == r.identifier; }
bool operator==(Point2D l, Point2D r) { return l.x == r.x && l.y == r.y; }

TEST(UnitTests, ProxyType) {
    auto dw = factory::default_wrapper<debug::vector, S, wrapper::layout::soa>(1);  // aos
    S<wrapper::value> s = {1, 2, {3.0, 4.0}, "Test"};
    dw[0].x = s.x; dw[0].y = s.y; dw[0].point = s.point; dw[0].identifier = s.identifier;

    S<wrapper::value> v = dw[0];
    S<wrapper::const_reference> cr = dw[0];
    S<wrapper::reference> r = dw[0];

    EXPECT_EQ(cr, s);
    EXPECT_EQ(r, s);

    int new_value = -1;
    EXPECT_NE(dw[0].x, new_value);
    r.x = new_value;
    EXPECT_EQ(dw[0].x, new_value);
}
