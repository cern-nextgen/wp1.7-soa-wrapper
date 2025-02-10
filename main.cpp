#include <iostream>

#include "kernel.h"


int main() {
    int error = kernel::run();
    std::cout << "Kernel output: " << error << std::endl;
    return 0;
}