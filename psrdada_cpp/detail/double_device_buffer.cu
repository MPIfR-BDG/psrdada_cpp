#include "psrdada_cpp/double_device_buffer.hpp"
#include <utility>

namespace psrdada_cpp {

template <typename T>
DoubleDeviceBuffer<T>::DoubleDeviceBuffer()
: _a_ptr(nullptr)
, _b_ptr(nullptr)
{
}

template <typename T>
DoubleDeviceBuffer<T>::~DoubleDeviceBuffer()
{

}

template <typename T>
void DoubleDeviceBuffer<T>::resize(std::size_t size, T fill_value)
{
    _buf0.resize(size, fill_value);
    _buf1.resize(size, fill_value);
    _a_ptr = thrust::raw_pointer_cast(_buf0);
    _b_ptr = thrust::raw_pointer_cast(_buf1);
}

template <typename T>
void DoubleDeviceBuffer<T>::swap()
{
    std::swap(_a_ptr, _b_ptr);
}

template <typename T>
T* DoubleDeviceBuffer<T>::a() const
{
    return _a_ptr;
}

template <typename T>
T* DoubleDeviceBuffer<T>::b() const
{
    return _b_ptr;
}

} //namespace psrdada_cpp