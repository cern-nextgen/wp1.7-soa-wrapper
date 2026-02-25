#ifndef DEBUG_H
#define DEBUG_H

#include <ostream>
#include <vector>
#include <span>

namespace debug {

struct counters {
    void reset() { *this = counters(); }
    bool operator==(const counters&) const = default;
    char default_constructor = 0;
    char nondefault_constructor = 0;
    char copy_constructor = 0;
    char copy_assignment = 0;
    char move_constructor = 0;
    char move_assignment = 0;
    char destructor = 0;
};

struct call_counter  {
    call_counter() { ++count.default_constructor; }
    call_counter(std::size_t size) { ++count.nondefault_constructor; }
    call_counter(const call_counter& other) { ++count.copy_constructor; }
    call_counter& operator=(const call_counter& other) {
        ++count.copy_assignment;
        return *this;
    }
    call_counter(call_counter&& other) { ++count.move_constructor; }
    call_counter& operator=(call_counter&& other) {
        ++count.move_assignment;
        return *this;
    }
    ~call_counter() { ++count.destructor; }
    static inline counters count;
    static void print(std::ostream& stream) {
        stream << "default constructor: " << call_counter::count.default_constructor << std::endl
               << "non-default constructor: " << call_counter::count.nondefault_constructor << std::endl
               << "copy constructor: " << call_counter::count.copy_constructor << std::endl
               << "copy assignment: " << call_counter::count.copy_assignment << std::endl
               << "move constructor: " << call_counter::count.move_constructor << std::endl
               << "move assignment: " << call_counter::count.move_assignment << std::endl
               << "destructor: " << call_counter::count.destructor << std::endl;
    }
};

template <class T>
struct vector : public std::vector<T>, public call_counter {
    vector() : std::vector<T>(), call_counter() {}
    vector(std::size_t size) : std::vector<T>(size), call_counter(size) {}
    constexpr operator std::span<T>() { return { this->data(), this->data() + this->size() }; }
};

}  // namespace debug

#endif  // DEBUG_H