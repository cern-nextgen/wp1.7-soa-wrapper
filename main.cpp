#include <iostream>
// #include <span>
// #include <vector>

#include "kernel.h"

// template <class T>
// using my_vector = std::vector<T>;  // avoid clang bug about default template parameters

int main() {

    int error = kernel::run();
    std::cout << "Kernel output: " << error << std::endl;

    return 0;
}