#include "allocator.h"
#include "wrapper.h"

#include <iostream>
#include <memory_resource>
#include <span>
#include <string>
#include <vector>

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
    constexpr std::size_t bytes = 1024;
    char buffer[bytes];

    /*allocator::BufferResource resource(buffer, bytes);
    std::pmr::vector<S<wrapper::value>> s_init = {
        {
            {0, 3, {0.0, 1.0}, "test"},
            {1, 4, {0.0, 1.0}, "foo"},
            {3, 5, {0.0, 1.0}, "bar"}
        },
        &resource
    };
    wrapper::wrapper<std::span, S, wrapper::layout::aos> my_array = {{s_init}};*/

    std::size_t interval = 10 * sizeof(int);
    allocator::BufferResource x_resource(buffer, interval);
    allocator::BufferResource y_resource(buffer + interval, interval);
    allocator::BufferResource point_resource(buffer + 2 * interval, 2 * interval);
    allocator::BufferResource identifier_resource(buffer + 4 * interval, bytes - 4 * interval);

    std::pmr::vector<int> x_init = {{0, 1, 2}, &x_resource};
    std::pmr::vector<int> y_init = {{3, 4, 5}, &y_resource};
    std::pmr::vector<Point2D> point_init = {{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}}, &point_resource};
    std::pmr::vector<std::string> identifier_init = {{"test", "foo", "bar"}, &identifier_resource};

    wrapper::wrapper<std::span, S, wrapper::layout::soa> my_array = {{x_init, y_init, point_init, identifier_init}};

    // reference
    for (int i = 0; i < 3; ++i) {
        S<wrapper::reference> r = my_array[i];
        r.setX(i - 10);
        r.y = i + 50;
        r.point = {0.5 * i, 0.5 * i};
        r.identifier = "foo" + std::to_string(i);
    }

    // const_reference
    for (int i = 0; i < 3; ++i) {
        S<wrapper::const_reference> cr = my_array[i];
        std::cout << "Element " << i << ": {"
                << cr.x << ", " << cr.y << ", {"
                << cr.point.x << ", " << cr.point.y << "}, "
                << cr.identifier << "}" << std::endl;
    }

    // member functions
    for (int i = 0; i < 3; ++i) {
        S<wrapper::value> v = my_array[i];
        std::cout << "my_array[" << i << "].getX()" << " == " << v.getX() << ", ";
        std::cout << "my_array[" << i << "].abs2() == "<< v.abs2() << std::endl;
    }

    return 0;
}