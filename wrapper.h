#ifndef WRAPPER_H
#define WRAPPER_H

namespace wrapper {

enum class layout { aos = 0, soa = 1 };

template<
    template <class> class F,
    template <template <class> class> class S,
    layout L
>
struct wrapper;

}  // namespace wrapper

#endif  // WRAPPER_H