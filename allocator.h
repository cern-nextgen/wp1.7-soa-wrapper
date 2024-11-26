#ifndef ALLOCATOR_H
#define ALLOCATOR_H

#include <memory>
#include <memory_resource>
#include <stdexcept>

namespace allocator {

struct BufferResource : public std::pmr::memory_resource {
    char* buffer_begin;
    char* buffer_end;
    char* current;

    BufferResource(char* buffer, std::size_t bytes)
        : buffer_begin(buffer), buffer_end(buffer + bytes), current(buffer) {}

protected:
    void* do_allocate(std::size_t bytes, std::size_t alignment) override {
        std::size_t space = buffer_end - current;
        void* aligned = std::align(alignment, bytes, reinterpret_cast<void*&>(current), space);
        if (!aligned) throw std::bad_alloc();
        current += bytes;
        return aligned;
    }

    void do_deallocate(void*, std::size_t, std::size_t) override {}

    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
        return this == &other;
    }
};

}  // namespace allocator

#endif  // ALLOCATOR_h