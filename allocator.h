#ifndef ALLOCATOR_H
#define ALLOCATOR_H

#include <cstddef>
#include <memory>

namespace allocator {

template <class T>
struct BufferAllocator {
    using value_type = T;

    BufferAllocator(char* buffer, std::size_t size)
        : buffer_(buffer), size_(size), offset_(0) {}

    T* allocate(std::size_t n) {
        std::size_t alignment = alignof(T);
        std::size_t bytes = n * sizeof(T);
        std::size_t aligned_offset = (offset_ + alignment - 1) & ~(alignment - 1);

        if (aligned_offset + bytes > size_) throw std::bad_alloc();

        T* ptr = std::launder(reinterpret_cast<T*>(buffer_ + aligned_offset));
        offset_ = aligned_offset + bytes;
        return ptr;
    }

    void deallocate(T*, std::size_t) { }

    template <typename U>
    struct rebind { using other = BufferAllocator<U>; };

    char* buffer_;
    std::size_t size_;
    std::size_t offset_;
};


}  // namespace allocator

#endif  // ALLOCATOR_H