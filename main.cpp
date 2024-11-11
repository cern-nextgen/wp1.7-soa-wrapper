#include "wrapper.h"

#include <iostream>
#include <string>
#include <vector>


struct Point2D { double x, y; };

template<template <class> class F>
struct S {
    F<int> x;
    F<int> y;
    F<Point2D> activation;
    F<std::string> identifier;

    int abs2() const { return x * x + y * y; }
    int& getX() { return x; }
    const int& getX() const { return x; }
    void setX(int x_new) { x = x_new; }
};

using wrapper_type = wrapper::wrapper<std::vector, S, wrapper::layout::soa>;

S<wrapper::identity> eval_const_at(const wrapper_type& w, std::size_t i) { return w[i]; }

int main() {
    wrapper_type my_array(3);

    // AoS access
    for (int i = 0; i < 3; ++i) {
        S<wrapper::reference> r = my_array[i];
        r.setX(i - 10);
        r.y = i + 50;
        r.activation = {0.5 * i, 0.5 * i};
        r.identifier = "foo" + std::to_string(i);
    }

    // const_reference
    for (int i = 0; i < 3; ++i) {
        S<wrapper::const_reference> r = my_array[i];
        std::cout << "Element " << i << ": {"
                << r.x << ", " << r.y << ", {"
                << r.activation.x << ", " << r.activation.y << "}, "
                << r.identifier << "}" << std::endl;
    }

    // member functions
    for (int i = 0; i < 3; ++i) {
        S<wrapper::identity> cr = eval_const_at(my_array, i);
        std::cout << "my_array[" << i << "].getX()" << " == " << cr.getX() << ", ";
        std::cout << "my_array[" << i << "].abs2() == "<< cr.abs2() << std::endl;
    }

    return 0;
}