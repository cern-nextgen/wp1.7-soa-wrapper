#include "helper.h"


namespace soa_wrapper {

template <typename T>
using identity = T;

template <typename T>
using reference = T&;

template <template <typename> typename F, template <template <typename> typename> typename S>
struct soa_wrapper : S<F> {
    using view_type = S<reference>;
    using data_type = S<F>;
    view_type operator[](std::size_t i) {
        auto f = [i](auto& v) -> decltype(auto) { return v[i]; };
        return helper::apply_to_members<data_type, view_type>(*this, f);
    }
};
    
}  // namespace aossoa
