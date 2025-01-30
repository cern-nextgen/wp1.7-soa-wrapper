#include "allocator.h"
#include "debug.h"
#include "factory.h"
#include "kernel.h"
#include "wrapper.h"

#include <iostream>
#include <span>
#include <vector>

struct Point2D { double x, y; };

template <template <class> class F>
struct S {
    F<int> x;
    F<int> y;
    F<Point2D> point;
    F<double> identifier;  // std::string causes linker error

    int abs2() const { return x * x + y * y; }
    int& getX() { return x; }
    const int& getX() const { return x; }
    void setX(int x_new) { x = x_new; }
};

template <class T>
using my_vector = std::vector<T>;  // avoid clang bug about default template parameters

int main() {
    constexpr std::size_t bytes = 1024;
    char buffer[bytes];

    // auto my_array = factory::buffer_wrapper<S, wrapper::layout::aos>(buffer, bytes);

    void runKernel();

    std::size_t N = 18;
    wrapper::wrapper<my_vector, S, wrapper::layout::soa> my_array{
        S<my_vector>{
            my_vector<int>(N),
            my_vector<int>(N),
            my_vector<Point2D>(N),
            my_vector<double>(N)
        }
    };

    // reference
    for (int i = 0; i < N; ++i) {
        S<wrapper::reference> r = my_array[i];
        r.setX(i - 10);
        r.y = i + 50;
        r.point = {0.5 * i, 0.5 * i};
        r.identifier = 0.1 * i;
    }

    // const_reference
    for (int i = 0; i < N; ++i) {
        S<wrapper::const_reference> cr = my_array[i];
        std::cout << "Element " << i << ": {"
                << cr.x << ", " << cr.y << ", {"
                << cr.point.x << ", " << cr.point.y << "}, "
                << cr.identifier << "}" << std::endl;
    }

    // member functions
    for (int i = 0; i < N; ++i) {
        S<wrapper::value> v = my_array[i];
        std::cout << "my_array[" << i << "].getX()" << " == " << v.getX() << ", ";
        std::cout << "my_array[" << i << "].abs2() == "<< v.abs2() << std::endl;
    }

    //debug::call_counter::print(std::cout);

    return 0;
}