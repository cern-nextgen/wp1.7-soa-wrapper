# memlayout

`memlayout` is a minimal header-only C++20 library designed to compile on:

- GCC
- Clang
- NVCC (CUDA)

## Features

- Header-only design
- __host__ / __device__ compatibility
- Unit tested with GoogleTest
- Documented with Doxygen

## Example

```cpp
#include <iostream>

#include <memlayout.h>

template <template <class> class Container>
struct Point {
    MEMLAYOUT_APPLY_UNARY(x, y, z)
    MEMLAYOUT_APPLY_BINARY(Point, MEMLAYOUT_EXPAND(x), MEMLAYOUT_EXPAND(y), MEMLAYOUT_EXPAND(z))
    Container<int> x, y, z;
};

template <template <class> class Container>
struct Momentum {
    MEMLAYOUT_APPLY_UNARY(p, i)
    MEMLAYOUT_APPLY_BINARY(Momentum, MEMLAYOUT_EXPAND(v), MEMLAYOUT_EXPAND(m))
    memlayout::Wrapper<Point, Container> v;
    Container<int> m;
};

int main()
{
    constexpr int N = 10;
    memlayout::Wrapper<Momentum, int[N], memlayout::Layout::soa> p{};
    for (int n = 0; n < N; ++n) {
        p[n].v.x = N - n;
        std::cout << p[n].v.x << ", ";
    }
    std::cout << std::endl;
    return 0;
}
```