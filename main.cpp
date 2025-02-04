#include <iostream>
// #include <span>
// #include <vector>

#include "allocator.h"
#include "debug.h"
#include "factory.h"
#include "kernel.h"
#include "skeleton.h"
#include "wrapper.h"

// template <class T>
// using my_vector = std::vector<T>;  // avoid clang bug about default template parameters

int main() {

    float error = kernel::run();
    std::cout << "Kernel output: " << error << std::endl;

    return 0;
}