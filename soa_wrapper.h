#ifndef SOA_WRAPPER_H
#define SOA_WRAPPER_H

#include "helper.h"


namespace soa_wrapper {

template <std::size_t, typename T>
using identity = T;

template <std::size_t, typename T>
using reference = T&;

template <template <std::size_t, class> typename F, template <template <std::size_t, class> class> class S>
struct soa_wrapper : S<F> {
    using view_type = S<reference>;
    using data_type = S<F>;

    template <class... Args>
    soa_wrapper(Args... args) {
        auto f = [args...](auto& member) -> decltype(auto) {
            member = std::remove_reference_t<decltype(member)>(args...);
            return member;
        };
        helper::apply_to_members<data_type, data_type>(*this, f);
    }

    view_type operator[](std::size_t i) {
        auto evaluate_at = [i](auto& member) -> decltype(auto) { return member[i]; };
        return helper::apply_to_members<data_type, view_type>(*this, evaluate_at);
    }
};

}  // namespace soa_wrapper

#endif  // SOA_WRAPPER_H