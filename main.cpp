#include <iostream>
#include <vector>
#include <span>

#include "skeleton.h"  // Defines the struct "S"
#include "wrapper.h"   // Contains the namespace "wrapper"

template <wrapper::layout L>
void print(std::ostream& os, wrapper::wrapper<S, std::span, L> w) {
    for (int i = 0; i < 2; ++i) {
        S<wrapper::const_reference> cr = w[i];
        os << "{" << cr.x << ", " << cr.y << ", {" << cr.point.x << ", " << cr.point.y << "}, " << cr.identifier << "}" << std::endl;
    }
}

int main() {
    wrapper::wrapper<S, std::vector, wrapper::layout::aos> w_aos{{
        {0, 1, {2.0, 3.0}, 4.0},
        {5, 6, {7.0, 8.0}, 9.0}
    }};

    wrapper::wrapper<S, std::vector, wrapper::layout::soa> w_soa{{
        {0, 5}, {1, 6}, {{2.0, 3.0}, {7.0, 8.0}}, {4.0, 9.0}
    }};

    print(std::cout, static_cast<wrapper::wrapper<S, std::span, wrapper::layout::soa>>(w_soa));  // Template argument deduction fails when passing w_aos or w_soa directly

    return 0;
}