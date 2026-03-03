#ifndef DEVICE_PTR_H
#define DEVICE_PTR_H

#include <span>

namespace test {

template <class T>
struct device_ptr {
    device_ptr(int N) : ptr(), N{N} { cudaMalloc((void **) &ptr, N * sizeof(T)); }
    ~device_ptr() { if (ptr != nullptr) cudaFree(ptr); }
    device_ptr(const device_ptr& other) = delete;
    device_ptr& operator=(const device_ptr& other) = delete;
    device_ptr(device_ptr&& other) noexcept : ptr(other.ptr), N(other.N) {
        other.ptr = nullptr;
        other.N = 0;
    }
    device_ptr& operator=(device_ptr&& other) noexcept {
        ptr = other.ptr;
        N = other.N;
        other.ptr = nullptr;
        other.N = 0;
        return *this;
    }
    constexpr operator std::span<T>() { return { ptr, ptr + N }; }
    constexpr T& operator[](int i) { return ptr[i]; }
    constexpr const T& operator[](int i) const { return ptr[i]; }
    constexpr T*& get() { return ptr; }

  private:
    T* ptr;
    int N;
};

}  // namespace test

#endif  // DEVICE_PTR_H