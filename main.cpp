#include <iostream>
#include <vector>
#include <span>

#include "skeleton.h"  // Defines the struct "S"
#include "wrapper.h"   // Contains the namespace "wrapper"

template <class T>
using my_vector = std::vector<T>;

template <class T>
using my_span = std::span<T>;

template <template <class> class F, template <template <class> class> class S, wrapper::layout L>
wrapper::wrapper<my_span, S, L> get_span(wrapper::wrapper<F, S, L> w) { return w; }

template <wrapper::layout L>
void print(std::ostream& os, wrapper::wrapper<my_span, S, L> w) {
    for (int i = 0; i < 2; ++i) {
        S<wrapper::const_reference> cr = w[i];
        os << "{" << cr.x << ", " << cr.y << ", {" << cr.point.x << ", " << cr.point.y << "}, " << cr.identifier << "}" << std::endl;
    }
}

int main() {
    wrapper::wrapper<my_vector, S, wrapper::layout::aos> w_aos{{
        {0, 1, {2.0, 3.0}, 4.0},
        {5, 6, {7.0, 8.0}, 9.0}
    }};

    wrapper::wrapper<my_vector, S, wrapper::layout::soa> w_soa{{
        {0, 5}, {1, 6}, {{2.0, 3.0}, {7.0, 8.0}}, {4.0, 9.0}
    }};

    print(std::cout, get_span(w_soa));  // Template argument deduction fails when passing w_aos or w_soa directly

    return 0;
}