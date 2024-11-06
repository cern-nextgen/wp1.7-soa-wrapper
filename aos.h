#ifndef AOS_H
#define AOS_H


#include <cstddef>

namespace aos {

template <typename T>
using identity = T;

template <template <class> class F, template <template <class> class> class S>
struct aos {
    using value_type = S<identity>;
    using array_type = F<value_type>;

    template <class... Args>
    aos(Args... args) : data(args...) { }

    array_type data;
    value_type& operator[](std::size_t i) { return data[i]; }
};
    
}  // namespace aos

#endif  // AOS_H
