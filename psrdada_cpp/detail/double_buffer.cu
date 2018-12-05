#include "psrdada_cpp/double_buffer.cuh"
#include <utility>

namespace psrdada_cpp {

template <typename T>
DoubleBuffer<T>::DoubleBuffer()
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
}

template <typename T>
void DoubleBuffer<T>::resize(std::size_t size, T fill_value)
{
    _buf0.resize(size, fill_value);
    _buf1.resize(size, fill_value);
}

template <typename T>
void DoubleBuffer<T>::swap()
{
    _buf0.swap(_buf1);
}

template <typename T>
typename DoubleBuffer<T>::VectorType& DoubleBuffer<T>::a()
{
    return _buf0;
}

template <typename T>
typename DoubleBuffer<T>::VectorType& DoubleBuffer<T>::b()
{
    return _buf1;
}

template <typename T>
T* DoubleBuffer<T>::a_ptr()
{
    return thrust::raw_pointer_cast(_buf0.data());
}

template <typename T>
T* DoubleBuffer<T>::b_ptr()
{
    return thrust::raw_pointer_cast(_buf1.data());
}

} //namespace psrdada_cpp
