#include "psrdada_cpp/double_device_buffer.hpp"
#include <utility>

namespace psrdada_cpp {

    template <typename T>
    DoubleBuffer<T>::DoubleBuffer()
    : _a_ptr(nullptr)
    , _b_ptr(nullptr)
    {
    }

    template <typename T>
    DoubleBuffer<T>::~DoubleBuffer()
    {

    }

    template <typename T>
    void DoubleBuffer<T>::resize(std::size_t size)
    {
        _buf0.resize(size);
        _buf1.resize(size);
        _a_ptr = thrust::raw_pointer_cast(_buf0);
        _b_ptr = thrust::raw_pointer_cast(_buf1);
    }

    template <typename T>
    void DoubleBuffer<T>::swap()
    {
        std::swap(_a_ptr, _b_ptr);
    }

    template <typename T>
    T* DoubleBuffer<T>::a() const
    {
        return _a_ptr;
    }

    template <typename T>
    T* DoubleBuffer<T>::b() const
    {
        return _b_ptr;
    }

} //namespace psrdada_cpp