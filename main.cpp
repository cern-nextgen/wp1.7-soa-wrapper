#include "aos.h"
#include "soa.h"
#include "wrapper.h"

#include <iostream>
#include <string>
#include <vector>


struct Point2D {
    double x, y;
};

template<template <std::size_t, class...> class F, class... Args>
struct S {
    F<0, int, Args...> x;
    F<1, int, Args...> y;
    F<2, Point2D, Args...> activation;
    F<3, std::string, Args...> identifier;

    int abs2() const requires wrapper::memberfunction_concept<F, S> { return x * x + y * y; }
    int& getX() requires wrapper::memberfunction_concept<F, S> { return x; }
    void setX(int x_new) requires wrapper::memberfunction_concept<F, S> { x = x_new; }
};

template <std::size_t, class T>
using my_vector = std::vector<T>;

int main() {
    wrapper::wrapper<my_vector, S, soa::soa> my_array(3);  // aos::aos

    // AoS access
    for (int i = 0; i < 3; ++i) {
        my_array[i].setX(i - 10);
        my_array[i].y = i + 50;
        my_array[i].activation = {0.5 * i, 0.5 * i};
        my_array[i].identifier = "foo" + std::to_string(i);
    }

    // SoA access
    my_array.y[1] = 42;
    my_array.activation[2].x *= 2;
    my_array.identifier[2] = "bla";

    for (int i = 0; i < 3; ++i) {
        std::cout << "Element " << i << ": {"
                << my_array[i].x << ", " << my_array[i].getX() << ", "
                << "{" << my_array[i].activation.x << ", " << my_array[i].activation.y << "}, "
                << my_array[i].identifier << "}" << std::endl;
    }

    std::cout << "my_array[1].abs2() == "<< my_array[1].abs2() << std::endl;
}