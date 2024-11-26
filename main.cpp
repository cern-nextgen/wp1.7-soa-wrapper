#include "wrapper.h"

#include <iostream>
#include <span>
#include <string>
#include <vector>


template <class T, std::size_t N>
struct raw_array {
    T t[N];
    T& operator[](std::size_t i) { return t[i]; }
    const T& operator[](std::size_t i) const { return t[i]; }
};

template <template <class, std::size_t> class array_type, std::size_t N>
struct bind_size {
    template <class T>
    using type = array_type<T, N>;
};

// bind_size<raw_array, 4>::type

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

template<wrapper::layout L>
constexpr wrapper::wrapper<std::span, S, L> construct_wrapper() {
    if constexpr (L == wrapper::layout::soa) {
        static std::vector<int> x_init{0, 1, 2};
        static std::vector<int> y_init{3, 4, 5};
        static std::vector<Point2D> point_init{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
        static std::vector<std::string> identifier_init{"test", "foo", "bar"};
        return {{x_init, y_init, point_init, identifier_init}};
    } else if constexpr (L == wrapper::layout::aos) {
        static std::vector<S<wrapper::value>> s_init = {
            {0, 3, {0.0, 1.0}, "test"},
            {1, 4, {0.0, 1.0}, "foo"},
            {3, 5, {0.0, 1.0}, "bar"}
        };
        return {{s_init}};
    }
};

int main() {
    auto my_array = construct_wrapper<wrapper::layout::aos>();

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