#include "psrdada_cpp/double_buffer.hpp"
#include <utility>

namespace psrdada_cpp {

    template <typename T>
    DoubleBuffer<T>::DoubleBuffer()
    {
        _a_ptr = &_buf0;
        _b_ptr = &_buf1;
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