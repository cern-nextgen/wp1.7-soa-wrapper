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
    void setX(int x_new) { x = x_new; }
};

int main() {
    wrapper::wrapper<std::vector, S, wrapper::soa> my_array(3);  // wrapper::aos

    // AoS access
    for (int i = 0; i < 3; ++i) {
        my_array[i].x = i - 10;
        my_array[i].y = i + 50;
        my_array[i].activation = {0.5 * i, 0.5 * i};
        my_array[i].identifier = "foo" + std::to_string(i);
    }

    // print
    for (int i = 0; i < 3; ++i) {
        std::cout << "Element " << i << ": {"
                << my_array[i].x << ", " << my_array[i].y << ", {"
                << my_array[i].activation.x << ", " << my_array[i].activation.y << "}, "
                << my_array[i].identifier << "}" << std::endl;
    }

    // member functions
    for (int i = 0; i < 3; ++i) {
        my_array[i].setX(42);
        std::cout << "my_array[" << i << "].getX()" << " == " << my_array[i].getX() << ", ";
        std::cout << "my_array[" << i << "].abs2() == "<< my_array[i].abs2() << std::endl;
    }

    return 0;
}