#include "soa_wrapper.h"
#include "aos_wrapper.h"

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
};

template <std::size_t, class T>
using my_vector = std::vector<T>;

int main() {
    // aos_wrapper::aos_wrapper<my_vector, S> my_array(3);
    soa_wrapper::soa_wrapper<my_vector, S> my_array(3);

    // AoS access
    for (int i = 0; i < 3; ++i) {
        my_array[i].x = i - 10;
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
                << my_array[i].x << ", " << my_array[i].y << ", "
                << "{" << my_array[i].activation.x << ", " << my_array[i].activation.y << "}, "
                << my_array[i].identifier << "}" << std::endl;
    }
}