#ifndef WRAPPER_SOA_H
#define WRAPPER_SOA_H


#include "helper.h"
#include "wrapper.h"

namespace wrapper {

template <typename T>
using reference = T&;

template <template <class> class F, template <template <class> class> class S>
struct wrapper<F, S, layout::soa> : public S<F> {
    using view_type = S<reference>;
    using data_type = S<F>;

    template <class... Args>
    wrapper(Args... args) {
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

}  // namespace wrapper

#endif  // WRAPPER_SOA_H