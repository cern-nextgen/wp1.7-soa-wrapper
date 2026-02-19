#include <iostream>
#include <vector>
#include <span>

#include "skeleton.h"  // Defines the struct "S"
#include "wrapper.h"

template <wrapper::layout L>
void print(std::ostream& os, wrapper::wrapper<skeleton::S, std::span, L> w) {
    if constexpr (L == wrapper::layout::aos) {
        os << "I am an AoS:" << std::endl;
    } else if constexpr (L == wrapper::layout::soa) {
        os << "I am an SoA:" << std::endl;
    } else {
        os << "I am don't know my layout:" << std::endl;
    }
    for (int i = 0; i < 2; ++i) {
        wrapper::wrapper<skeleton::S, wrapper::const_reference> cr = w[i];
        os << "{" << cr.x << ", " << cr.y << ", {" << cr.point.x << ", " << cr.point.y << "}, " << cr.identifier << "}" << std::endl;
    }
}

//template void print<wrapper::layout::aos>(std::ostream& os, wrapper::wrapper<skeleton::S, std::span, wrapper::layout::aos> w);
//template void print<wrapper::layout::soa>(std::ostream& os, wrapper::wrapper<skeleton::S, std::span, wrapper::layout::soa> w);

/*void print(std::ostream& os, wrapper::wrapper<skeleton::S, std::span, wrapper::layout::aos> w) {
    os << "I am an AoS:" << std::endl;
    for (int i = 0; i < 2; ++i) {
        wrapper::wrapper<skeleton::S, wrapper::const_reference> cr = w[i];
        os << "{" << cr.x << ", " << cr.y << ", {" << cr.point.x << ", " << cr.point.y << "}, " << cr.identifier << "}" << std::endl;
    }
}

void print(std::ostream& os, wrapper::wrapper<skeleton::S, std::span, wrapper::layout::soa> w) {
    os << "I am an SoA:" << std::endl;
    for (int i = 0; i < 2; ++i) {
        wrapper::wrapper<skeleton::S, wrapper::const_reference> cr = w[i];
        os << "{" << cr.x << ", " << cr.y << ", {" << cr.point.x << ", " << cr.point.y << "}, " << cr.identifier << "}" << std::endl;
    }
}*/

using test = wrapper::wrapper<skeleton::S, std::span, wrapper::layout::aos>;

int main() {
    wrapper::wrapper<skeleton::S, std::vector, wrapper::layout::aos> w_aos{{
        {0, 1, {2.0, 3.0}, 4.0},
        {5, 6, {7.0, 8.0}, 9.0}
    }};
    wrapper::wrapper<skeleton::S, std::span, wrapper::layout::aos> w_aos_span{w_aos};

    wrapper::wrapper<skeleton::S, std::vector, wrapper::layout::soa> w_soa{{
        {0, 5}, {1, 6}, {{2.0, 3.0}, {7.0, 8.0}}, {4.0, 9.0}
    }};
    wrapper::wrapper<skeleton::S, std::span, wrapper::layout::soa> w_soa_span{w_soa};

    print<w_aos.layout_type>(std::cout, w_aos);
    print<w_soa.layout_type>(std::cout, w_soa);

    std::cout << sizeof(w_aos) << " " << sizeof(w_soa) << std::endl;

    return 0;
}