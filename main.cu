#include <iostream>
#include <vector>
#include <span>

#include "skeleton.h"
#include "wrapper.h"

template <wrapper::layout L>
void print(std::ostream& os, wrapper::wrapper<Skeleton::Point2D, std::span, L> w) {
    if constexpr (L == wrapper::layout::aos) {
        os << "I am an AoS: ";
    } else if constexpr (L == wrapper::layout::soa) {
        os << "I am an SoA: ";
    } else {
        os << "I am don't know my layout: ";
    }
    os << "{" << w[0].x << ", " << w[0].y << "}";
    for (int i = 1; i < 5; ++i) {
        wrapper::wrapper<Skeleton::Point2D, wrapper::const_reference> cr = w[i];
        os << ", {" << cr.x << ", " << cr.y << "}";
    }
    os << std::endl;
}

int main() {
    wrapper::wrapper<Skeleton::Point2D, std::vector, wrapper::layout::aos> w_aos{
        {0, 5}, {1, 6}, {2, 7}, {3, 8}, {4, 9}
    };
    wrapper::wrapper<Skeleton::Point2D, std::span, wrapper::layout::aos> w_aos_span{w_aos};

    wrapper::wrapper<Skeleton::Point2D, std::vector, wrapper::layout::soa> w_soa{
        {{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}}
    };
    wrapper::wrapper<Skeleton::Point2D, std::span, wrapper::layout::soa> w_soa_span{w_soa};

    print<w_aos.layout_type>(std::cout, w_aos);
    print<w_soa.layout_type>(std::cout, w_soa);

    std::cout << "sizeof aos: " << sizeof(w_aos) << ", sizeof soa: " << sizeof(w_soa) << std::endl;

    return 0;
}