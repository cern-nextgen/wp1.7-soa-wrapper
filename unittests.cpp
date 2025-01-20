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

template<template <class> class F, wrapper::layout L>
void proxy_type(std::size_t N, wrapper::wrapper<F, S, L> w) {
    // reference
    for (int i = 0; i < N; ++i) {
        S<wrapper::reference> r = w[i];

        r.setX(-1);
        EXPECT_EQ(w[i].x, -1);

        r.getX() = i - 10;
        r.y = i + 50;
        r.point = {0.5 * i, 0.5 * i};
        r.identifier = "Bla";

        EXPECT_EQ(w[i].x, i - 10);
        EXPECT_EQ(w[i].y, i + 50);
        EXPECT_EQ(w[i].point, Point2D(0.5 * i, 0.5 * i));
        EXPECT_EQ(w[i].identifier, "Bla");
    }

    wrapper::wrapper<F, S, L> w_copy = w;

    // const_reference
    for (int i = 0; i < N; ++i) {
        S<wrapper::const_reference> cr = w[i];

        EXPECT_EQ(cr.abs2(), w_copy[i].abs2());

        EXPECT_EQ(cr.x, w_copy[i].x);
        EXPECT_EQ(cr.y, w_copy[i].y);
        EXPECT_EQ(cr.point, w_copy[i].point);
        EXPECT_EQ(cr.identifier, w_copy[i].identifier);
    }

    // value
    for (int i = 0; i < N; ++i) {
        S<wrapper::value> v = w[i];

        v.setX(i - 1);
        v.y = i + 5;
        v.point = {5.0 * i, 5.0 * i};
        v.identifier = "Test";

        EXPECT_EQ(w[i].getX(), w_copy[i].getX());
        EXPECT_EQ(w[i].abs2(), w_copy[i].abs2());

        EXPECT_EQ(w[i].x, w_copy[i].x);
        EXPECT_EQ(w[i].y, w_copy[i].y);
        EXPECT_EQ(w[i].point, w_copy[i].point);
        EXPECT_EQ(w[i].identifier, w_copy[i].identifier);
    }
}

TEST(ProxyType, AoS) {
    std::size_t N = 5;
    auto w = factory::default_wrapper<debug::vector, S, wrapper::layout::aos>(N);
    proxy_type<debug::vector, wrapper::layout::aos>(N, std::move(w));
}
TEST(ProxyType, SoA) {
    std::size_t N = 5;
    auto w = factory::default_wrapper<debug::vector, S, wrapper::layout::soa>(N);
    proxy_type<debug::vector, wrapper::layout::soa>(N, std::move(w));
}
