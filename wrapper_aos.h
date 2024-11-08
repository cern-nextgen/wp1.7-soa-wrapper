#ifndef WRAPPER_AOS_H
#define WRAPPER_AOS_H


#include "wrapper.h"

#include <cstddef>

namespace wrapper {

template <typename T>
using identity = T;

template <template <class> class F, template <template <class> class> class S>
struct wrapper<F, S, layout::aos>  {
    using value_type = S<identity>;
    using array_type = F<value_type>;

    template <class... Args>
    wrapper(Args... args) : data(args...) { }

    array_type data;
    value_type& operator[](std::size_t i) { return data[i]; }
};
    
}  // namespace wrapper

#endif  // WRAPPER_AOS_H
