#include "allocator.h"
#include "debug.h"
#include "factory.h"
#include "wrapper.h"

#include <iostream>
#include <span>
#include <string>

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

int main() {
    {
    constexpr std::size_t bytes = 1024;
    char buffer[bytes];

    //auto my_array = factory::make_wrapper<S, wrapper::layout::aos>(buffer, bytes);

    std::size_t N = 18;
    wrapper::wrapper<debug::vector, S, wrapper::layout::soa> my_array{
        S<debug::vector>{
            debug::vector<int>(N),
            debug::vector<int>(N),
            debug::vector<Point2D>(N),
            debug::vector<std::string>(N)
        }
    };

    // reference
    for (int i = 0; i < N; ++i) {
        S<wrapper::reference> r = my_array[i];
        r.setX(i - 10);
        r.y = i + 50;
        r.point = {0.5 * i, 0.5 * i};
        r.identifier = "Bla";
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
    }

    debug::call_counter::print(std::cout);

    return 0;
}