#include <vector>
#include <span>

#include <gtest/gtest.h>

#include "skeleton.h"
#include "wrapper.h"

template <wrapper::layout L>
uint32_t compute_sum(int N, wrapper::wrapper<Skeleton::Point2D, std::span, L> w) {
    uint32_t sum = 0;
    for (int i = 0; i < N; ++i) {
        wrapper::wrapper<Skeleton::Point2D, wrapper::const_reference> cr = w[i];
        sum += cr.x + cr.y;
    }
    return sum;
}

TEST(OverloadResolution, AoS) {
    wrapper::wrapper<Skeleton::Point2D, std::vector, wrapper::layout::aos> w{
        {0, 5}, {1, 6}, {2, 7}, {3, 8}, {4, 9}
    };
    uint32_t sum = compute_sum<w.layout_type>(5, w);
    uint32_t expected_sum = 45;
    EXPECT_EQ(expected_sum, sum);
}
TEST(OverloadResolution, SoA) {
    wrapper::wrapper<Skeleton::Point2D, std::vector, wrapper::layout::soa> w{
        {{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}}
    };
    uint32_t sum = compute_sum<w.layout_type>(5, w);
    uint32_t expected_sum = 45;
    EXPECT_EQ(expected_sum, sum);
}

TEST(VectorToSpan, AoS) {
    wrapper::wrapper<Skeleton::Point2D, std::vector, wrapper::layout::aos> w{
        {0, 5}, {1, 6}, {2, 7}, {3, 8}, {4, 9}
    };
    wrapper::wrapper<Skeleton::Point2D, std::span, wrapper::layout::aos> w_span{w};
    uint32_t sum = compute_sum(5, w_span);
    uint32_t expected_sum = 45;
    EXPECT_EQ(expected_sum, sum);
}
TEST(VectorToSpan, SoA) {
    wrapper::wrapper<Skeleton::Point2D, std::vector, wrapper::layout::soa> w{
        {{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}}
    };
    wrapper::wrapper<Skeleton::Point2D, std::span, wrapper::layout::soa> w_span{w};
    uint32_t sum = compute_sum(5, w_span);
    uint32_t expected_sum = 45;
    EXPECT_EQ(expected_sum, sum);
}
