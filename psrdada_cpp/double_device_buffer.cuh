#ifndef PSRDADA_CPP_DOUBLE_DEVICE_BUFFER_HPP
#define PSRDADA_CPP_DOUBLE_DEVICE_BUFFER_HPP

#include "thrust/device_vector.h"

namespace psrdada_cpp {

template <typename T>
class DoubleDeviceBuffer
{
public:
    typedef thrust::device_vector<T> VectorType;

public:
    DoubleDeviceBuffer();
    ~DoubleDeviceBuffer();
    void resize(std::size_t size);
    void resize(std::size_t size, T fill_value);
    void swap();

    VectorType& a() const;
    VectorType& b() const;
    T* a_ptr() const;
    T* b_ptr() const;

private:
    VectorType _buf0;
    VectorType _buf1;
};

} //namespace psrdada_cpp

#include "psrdada_cpp/detail/double_device_buffer.cu"

#endif //PSRDADA_CPP_DOUBLE_DEVICE_BUFFER_HPP
