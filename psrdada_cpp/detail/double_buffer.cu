#include "psrdada_cpp/double_buffer.cuh"
#include <utility>

namespace psrdada_cpp {

template <typename VectorType>
DoubleBuffer<VectorType>::DoubleBuffer()
{
}

template <typename VectorType>
DoubleBuffer<VectorType>::~DoubleBuffer()
{

}

template <typename VectorType>
void DoubleBuffer<VectorType>::resize(std::size_t size)
{
    _buf0.resize(size);
    _buf1.resize(size);
}

template <typename VectorType>
void DoubleBuffer<VectorType>::resize(std::size_t size, typename VectorType::value_type fill_value)
{
    _buf0.resize(size, fill_value);
    _buf1.resize(size, fill_value);
}

template <typename VectorType>
void DoubleBuffer<VectorType>::swap()
{
    _buf0.swap(_buf1);
}

template <typename VectorType>
VectorType& DoubleBuffer<VectorType>::a()
{
    return _buf0;
}

template <typename VectorType>
VectorType& DoubleBuffer<VectorType>::b()
{
    return _buf1;
}

template <typename VectorType>
typename VectorType::value_type* DoubleBuffer<VectorType>::a_ptr()
{
    return thrust::raw_pointer_cast(_buf0.data());
}

template <typename VectorType>
typename VectorType::value_type* DoubleBuffer<VectorType>::b_ptr()
{
    return thrust::raw_pointer_cast(_buf1.data());
}

} //namespace psrdada_cpp
