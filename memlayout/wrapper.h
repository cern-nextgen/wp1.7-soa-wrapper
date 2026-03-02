#ifndef WRAPPER_H
#define WRAPPER_H

namespace memlayout {

using size_t = decltype(sizeof(0));
using ptrdiff_t = decltype(static_cast<int*>(nullptr) - static_cast<int*>(nullptr));

template <class T> using value = T;
template <class T> using reference = T&;
template <class T> using const_reference = const T&;
template <class T> using pointer = T*;
template <class T> using const_pointer = const T*;

template <class ReturnType>
struct RandomAccessAt {
    memlayout::size_t i;
    template <class... Args>
    constexpr ReturnType operator()(Args& ...args) const { return {args[i]...}; }
    template <class... Args>
    constexpr ReturnType operator()(const Args& ...args) const { return {args[i]...}; }
};

template <class ReturnType>
struct GetPointer {
    template <class... Args>
    constexpr ReturnType operator()(Args& ...args) const { return {&args...}; }
    template <class... Args>
    constexpr ReturnType operator()(const Args& ...args) const { return {&args...}; }
};

template <class ReturnType>
struct AggregateConstructor {
    template <class... Args>
    constexpr ReturnType operator()(Args& ...args) const { return {args...}; }
    template <class... Args>
    constexpr ReturnType operator()(const Args& ...args) const { return {args...}; }
};

struct FirstMember {
    template <class T, class... Args>
    constexpr T& operator()(T& t, Args& ...args) const { return t; }
    template <class T, class... Args>
    constexpr const T& operator()(const T& t, const Args& ...args) const { return t; }
};

template <class ReturnType>
struct PreIncrement {
    template <class... Args>
    constexpr ReturnType operator()(Args& ...args) const { return {++args...}; }
};

template <class ReturnType>
struct PreDecrement {
    template <class... Args>
    constexpr ReturnType operator()(Args& ...args) const { return {--args...}; }
};

template <class ReturnType>
struct Advance {
    ptrdiff_t i;
    template <class... Args>
    constexpr ReturnType operator()(const Args& ...args) const { return {(args + i)...}; }
};

struct CopyAssignment {
    template <class Left, class Right>
    constexpr Left& operator()(Left& left, const Right& right) const { return left = right; }
};

enum class Layout { aos = 0, soa = 1 };

template <
    template <template <class> class> class Struct,
    template <class> class Container,
    Layout L = Layout::soa
>
struct Wrapper;

template <
    template <template <class> class> class Struct,
    template <class> class Container
>
struct Wrapper<Struct, Container, Layout::aos> : public Container<Struct<value>> {
    static constexpr Layout layout_type = Layout::aos;
    using Base = Container<Struct<value>>;
    using Base::Base;

    constexpr Wrapper() = default;
    template <template <class> class other_Container>
    constexpr Wrapper(Wrapper<Struct, other_Container, Layout::aos>& other) : Base(other) {}

    constexpr Wrapper<Struct, reference> operator[] (size_t i) { return Base::operator[](i); }
    constexpr Wrapper<Struct, const_reference> operator[] (size_t i) const { return Base::operator[](i); }

    constexpr Wrapper<Struct, reference> operator*() { return operator[](0); }
    constexpr Wrapper<Struct, const_reference> operator*(ptrdiff_t) const { return operator[](0); }
};

template <template <template <class> class> class Struct>
struct Wrapper<Struct, pointer, Layout::aos> {
    static constexpr Layout layout_type = Layout::aos;

    pointer<Struct<value>> data;
    using Data = pointer<Struct<value>>;

    operator Data&() { return data; }
    operator const Data&() const { return data; }

    constexpr bool operator==(const Wrapper& other) const { return data == other.data; }
    constexpr bool operator!=(const Wrapper& other) const { return data != other.data; }
    constexpr bool operator<(const Wrapper& other) const { return data < other.data; }

    constexpr Wrapper operator+(ptrdiff_t i) const { return {data + i}; }
    constexpr Wrapper operator-(ptrdiff_t i) const { return {data - i}; }
    constexpr ptrdiff_t operator-(const Wrapper& other) const { return {data - other.data}; }

    constexpr Wrapper& operator++() { return {++data}; }
    constexpr Wrapper& operator+=(ptrdiff_t i) { return {data += i}; }
    constexpr Wrapper& operator--() { return {--data}; }
    constexpr Wrapper& operator-=(ptrdiff_t i) { return {data -= i}; }

    constexpr Wrapper<Struct, reference> operator[] (size_t i) { return data[i]; }
    constexpr Wrapper<Struct, const_reference> operator[] (size_t i) const { return data[i]; }

    constexpr Wrapper<Struct, reference> operator*() { return *data; }
    constexpr Wrapper<Struct, const_reference> operator*(ptrdiff_t) const { return *data; }
};

template <
    template <template <class> class> class Struct,
    template <class> class Container
>
struct Wrapper<Struct, Container, Layout::soa> : public Struct<Container> {
    static constexpr Layout layout_type = Layout::soa;
    using Base = Struct<Container>;
    using Base::Base;

    constexpr Wrapper() = default;
    constexpr Wrapper(Base b) : Base{static_cast<Base&&>(b)} {}
    template <template <class> class other_Container>
    constexpr Wrapper(Struct<other_Container>& other) : Base{other.apply(AggregateConstructor<Base>{})} {}
    template <template <class> class other_Container>
    constexpr Wrapper(const Struct<other_Container>& other) : Base{other.apply(AggregateConstructor<Base>{})} {}

    constexpr Wrapper<Struct, reference> operator[] (size_t i) { return Base::apply(RandomAccessAt<Struct<reference>>{i}); }
    constexpr Wrapper<Struct, const_reference> operator[] (size_t i) const { return Base::apply(RandomAccessAt<Struct<const_reference>>{i}); }

    constexpr Wrapper<Struct, reference> operator*() { return operator[](0); }
    constexpr Wrapper<Struct, const_reference> operator*(ptrdiff_t) const { return operator[](0); }
};

template <template <template <class> class> class Struct>
struct Wrapper<Struct, value, Layout::soa> : public Struct<value> {
    using Base = Struct<value>;

    constexpr Wrapper() = default;
    constexpr Wrapper(Base b) : Base{static_cast<Base&&>(b)} {}
    constexpr Wrapper(const Struct<reference>& other) : Base(other.apply(AggregateConstructor<Base>{})) {}
    constexpr Wrapper(const Struct<const_reference>& other) : Base(other.apply(AggregateConstructor<Base>{})) {}
};

template <template <template <class> class> class Struct>
struct Wrapper<Struct, reference, Layout::soa> : public Struct<reference> {
    using Base = Struct<reference>;

    constexpr Wrapper() = delete;
    constexpr Wrapper(Base b) : Base{static_cast<Base&&>(b)} {}
    constexpr Wrapper(Struct<value>& other) : Base(other.apply(AggregateConstructor<Base>{})) {}
    
    constexpr Wrapper(const Wrapper& other) = default;

    constexpr Wrapper& operator=(const Wrapper<Struct, value>& other) {
        Base::apply(other, CopyAssignment{});
        return *this;
    }
    constexpr Wrapper& operator=(const Wrapper& other) {
        Base::apply(other, CopyAssignment{});
        return *this;
    }
    constexpr Wrapper& operator=(const Wrapper<Struct, const_reference>& other) {
        Base::apply(other, CopyAssignment{});
        return *this;
    }

    constexpr Wrapper(Wrapper&& other) = default;
    
    constexpr Wrapper& operator=(Wrapper&& other) { return operator=(other); }

    constexpr Wrapper<Struct, pointer> operator&() { return Base::apply(GetPointer<Struct<pointer>>{}); }
    //constexpr Wrapper<Struct, const_pointer> operator&() const { return Base::apply(GetPointer<Struct<const_pointer>>{}); }
    constexpr pointer<Wrapper<Struct, reference>> operator->() { return this; }
};

template <template <template <class> class> class Struct>
struct Wrapper<Struct, const_reference, Layout::soa> : public Struct<const_reference> {
    using Base = Struct<const_reference>;

    constexpr Wrapper() = delete;
    constexpr Wrapper(Base b) : Base{static_cast<Base&&>(b)} {}
    constexpr Wrapper(const Struct<value>& other) : Base(other.apply(AggregateConstructor<Base>{})) {}
    constexpr Wrapper(const Struct<reference>& other) : Base(other.apply(AggregateConstructor<Base>{})) {}

    constexpr Wrapper<Struct, const_pointer> operator&() const { return Base::apply(GetPointer<Struct<const_pointer>>{}); }
    constexpr const_pointer<Wrapper<Struct, const_reference>> operator->() const { return this; }
};

template <template <template <class> class> class Struct>
struct Wrapper<Struct, pointer, Layout::soa> : public Struct<pointer> {
    using Base = Struct<pointer>;

    constexpr Wrapper() = default;
    constexpr Wrapper(Base b) : Base{static_cast<Base&&>(b)} {}

    constexpr Wrapper<Struct, reference> operator[] (size_t i) { return Base::apply(RandomAccessAt<Struct<reference>>{i}); }
    constexpr const Wrapper<Struct, const_reference> operator[] (size_t i) const { return Base::apply(RandomAccessAt<Struct<const_reference>>{i}); }

    constexpr Wrapper<Struct, reference> operator*() { return operator[](0); }
    constexpr Wrapper<Struct, const_reference> operator*() const { return operator[](0); }
    constexpr Wrapper<Struct, reference> operator->() { return operator[](0); }
    constexpr Wrapper<Struct, const_reference> operator->() const { return operator[](0); }

    constexpr bool operator==(const Wrapper& other) const { return Base::apply(FirstMember{}) == other.apply(FirstMember{}); }
    constexpr bool operator!=(const Wrapper& other) const { return !this->operator==(other); }
    constexpr bool operator<(const Wrapper& other) const { return Base::apply(FirstMember{}) < other.apply(FirstMember{}); }

    constexpr Wrapper operator+(ptrdiff_t i) const { return Base::apply(Advance<Base>{i}); }
    constexpr Wrapper operator-(ptrdiff_t i) const { return operator+(-i); }
    constexpr ptrdiff_t operator-(const Wrapper& other) const { return Base::apply(FirstMember{}) - other.apply(FirstMember{}); }

    constexpr Wrapper& operator++() { Base::apply(PreIncrement<Base>{}); return *this; }
    constexpr Wrapper& operator+=(ptrdiff_t i) { return *this = *this + i; }
    constexpr Wrapper& operator--() { Base::apply(PreDecrement<Base>{}); return *this; }
    constexpr Wrapper& operator-=(ptrdiff_t i) { return *this = *this - i; }
};

template <template <template <class> class> class Struct>
struct Wrapper<Struct, const_pointer, Layout::soa> : public Struct<const_pointer> {
    using Base = Struct<const_pointer>;

    constexpr Wrapper() = default;
    constexpr Wrapper(Base b) : Base{static_cast<Base&&>(b)} {}
    constexpr Wrapper(const Struct<pointer>& other) : Base(other.apply(AggregateConstructor<Base>{})) {}

    constexpr Wrapper<Struct, const_reference> operator[] (size_t i) const { return Base::apply(RandomAccessAt<Struct<const_reference>>{i}); }
    constexpr Wrapper<Struct, const_reference> operator*() const { return operator[](0); }
    constexpr Wrapper<Struct, const_reference> operator->() const { return operator[](0); }

    constexpr bool operator==(const Wrapper& other) const { return Base::apply(FirstMember{}) == other.apply(FirstMember{}); }
    constexpr bool operator!=(const Wrapper& other) const { return !this->operator==(other); }
    constexpr bool operator<(const Wrapper& other) const { return Base::apply(FirstMember{}) < other.apply(FirstMember{}); }

    constexpr Wrapper operator+(ptrdiff_t i) const { return Base::apply(Advance<Base>{i}); }
    constexpr Wrapper operator-(ptrdiff_t i) const { return operator+(-i); }
    constexpr ptrdiff_t operator-(const Wrapper& other) const { return Base::apply(FirstMember{}) - other.apply(FirstMember{}); }

    constexpr Wrapper& operator++() { Base::apply(PreIncrement<Base>{}); return *this; }
    constexpr Wrapper& operator+=(ptrdiff_t i) { return *this = *this + i; }
    constexpr Wrapper& operator--() { Base::apply(PreDecrement<Base>{}); return *this; }
    constexpr Wrapper& operator-=(ptrdiff_t i) { return *this = *this - i; }
};

}  // namespace Wrapper

#define WRAPPER_APPLY_UNARY(...)\
    template <class Function>\
    constexpr auto apply(Function&& f) { return f(__VA_ARGS__); }\
    template <class Function>\
    constexpr auto apply(Function&& f) const { return f(__VA_ARGS__); }\

#define WRAPPER_EXPAND(m) f(m, other.m)

#define WRAPPER_APPLY_BINARY(STRUCT_NAME, ...)\
    template <template <class> class other_Container, class Function>\
    constexpr STRUCT_NAME apply(STRUCT_NAME<other_Container>& other, Function&& f) { return {__VA_ARGS__}; }\
    template <template <class> class other_Container, class Function>\
    constexpr STRUCT_NAME apply(STRUCT_NAME<other_Container>& other, Function&& f) const { return {__VA_ARGS__}; }\
    template <template <class> class other_Container, class Function>\
    constexpr STRUCT_NAME apply(const STRUCT_NAME<other_Container>& other, Function&& f) { return {__VA_ARGS__}; }\
    template <template <class> class other_Container, class Function>\
    constexpr STRUCT_NAME apply(const STRUCT_NAME<other_Container>& other, Function&& f) const { return {__VA_ARGS__}; }\

#endif  // WRAPPER_H