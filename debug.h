#ifndef DEBUG_H
#define DEBUG_H

#include <ostream>
#include <vector>

namespace debug {

struct call_counter  {
    call_counter() { ++default_constructor; }
    call_counter(std::size_t size) { ++constructor; }
    call_counter(const call_counter& other) { ++copy_constructor; }
    call_counter& operator=(const call_counter& other) {
        ++copy_assignment;
        return *this;
    }
    call_counter(call_counter&& other) { ++move_constructor; }
    call_counter& operator=(call_counter&& other) {
        ++move_assignment;
        return *this;
    }
    ~call_counter() { ++destructor; }
    static inline std::size_t default_constructor = 0;
    static inline std::size_t constructor = 0;
    static inline std::size_t copy_constructor = 0;
    static inline std::size_t copy_assignment = 0;
    static inline std::size_t move_constructor = 0;
    static inline std::size_t move_assignment = 0;
    static inline std::size_t destructor = 0;
    static void print(std::ostream& stream) {
        stream << "default constructor: " << call_counter::default_constructor << std::endl
               << "constructor: " << call_counter::constructor << std::endl
               << "copy constructor: " << call_counter::copy_constructor << std::endl
               << "copy assignment: " << call_counter::copy_assignment << std::endl
               << "move constructor: " << call_counter::move_constructor << std::endl
               << "move assignment: " << call_counter::move_assignment << std::endl
               << "destructor: " << call_counter::destructor << std::endl;
    }
};

template <class T>
struct vector : public std::vector<T>, public call_counter {
    vector() : std::vector<T>(), call_counter() {}
    vector(std::size_t size) : std::vector<T>(size), call_counter(size) {}
};

}  // namespace debug

#endif  // DEBUG_H