#include "soa_wrapper.h"

#include <iostream>
#include <vector>


template <template <typename> typename F>
struct S {
    F<int> x, y;
    F<double> activation;
    F<int> identifier;
};

int main() {
    soa_wrapper::soa_wrapper<std::vector, S> my_array;
    my_array.x = {0, 1, 2, 3};
    my_array.y = {0, -1, -2, -3};
    my_array.activation = {0.0, 1.0, 2.0, 3.0};
    my_array.identifier = {0, 1, 2, 3};

    for (int i = 0; i < 4; ++i) {
        auto e = my_array[i];
        e.x = i - 10;
        e.y = i + 50;
        e.activation = 0.5 * i;
        e.identifier = i;
    }

    my_array.y[2] = 42;

    for (int i = 0; i < 4; ++i) {
        auto e = my_array[i];
        std::cout << "Element " << i << ": {" << e.x << ", " << e.y << ", " << e.activation << ", " << e.identifier << "}" << std::endl;
    }
}