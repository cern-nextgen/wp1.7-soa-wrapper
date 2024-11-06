#ifndef AOS_H
#define AOS_H

#include "helper.h"

#include <tuple>


namespace aos {

template <std::size_t, typename T>
using identity = T;

template <std::size_t I, class T>
struct Select {
    template <class... Args>
    Select(Args&... args) : get{std::get<I>(std::tie(args...))} { }
    T& get;
};

template <std::size_t I, class T, class base_type>
struct soa_view {
    base_type* data_ptr;
    T& operator[](std::size_t i) {
        auto id = [](auto& member) -> decltype(auto) { return member; };
        Select s = helper::apply_to_members<typename base_type::value_type, Select<I, T>>(data_ptr->data[i], id);
        return s.get;
    }
};

template <template <std::size_t, class...> class F, template <template <std::size_t, class...> class, class...> class S>
struct aos : public S<soa_view, aos<F, S>> {
    using base_type = S<soa_view, aos<F, S>>;
    using value_type = S<identity>;
    using array_type = F<0, value_type>;

    template <class... Args>
    aos(Args... args) : data(args...) {
        auto initialize = [this](auto& member) -> decltype(auto) { member.data_ptr = this; return member; };
        helper::apply_to_members<base_type, base_type>(*static_cast<base_type*>(this), initialize);
    }

    array_type data;
    value_type& operator[](std::size_t i) { return data[i]; }
};
    
}  // namespace aos

#endif  // AOS_H
