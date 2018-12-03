#ifndef PSRDADA_CPP_DOUBLE_DEVICE_BUFFER_HPP
#define PSRDADA_CPP_DOUBLE_DEVICE_BUFFER_HPP

#include "thrust/device_vector.h"

namespace psrdada_cpp {

/**
 * @brief      Class for double device buffer.
 *
 * @tparam     T     The internal data type for the buffers
 *
 * @detail     An implementation of the double buffer concept
 *             using thrust::device_vector. Provides double
 *             buffers in GPU memory.
 *
 *             This class exposes two buffers "a" and "b"
 *             which can be independently accessed. The buffers
 *             can be swapped to make b=a and a=b which is useful
 *             for staging inputs and outputs in a streaming pipeline
 */
template <typename T>
class DoubleDeviceBuffer
{
public:
    typedef thrust::device_vector<T> VectorType;

public:
    /**
     * @brief      Constructs the object.
     */
    DoubleDeviceBuffer();
    ~DoubleDeviceBuffer();
    DoubleDeviceBuffer(DoubleDeviceBuffer const&) = delete;

    /**
     * @brief      Resize the buffer
     *
     * @param[in]  size  The desired size in units of the data type
     */
    void resize(std::size_t size);

    /**
     * @brief      Resize the buffer
     *
     * @param[in]  size        The desired size in units of the data type
     * @param[in]  fill_value  The fill value for newly allocated memory
     */
    void resize(std::size_t size, T fill_value);

    /**
     * @brief      Swap the a and b buffers
     */
    void swap();

    /**
     * @brief      Return a reference to the "a" vector
     */
    VectorType& a();

    /**
     * @brief      Return a reference to the "b" vector
     */
    VectorType& b();

    /**
     * @brief      Return a pointer to the contents of the "a" vector
     */
    T* a_ptr();

    /**
     * @brief      Return a pointer to the contents of the "b" vector
     */
    T* b_ptr();

private:
    VectorType _buf0;
    VectorType _buf1;
};

} //namespace psrdada_cpp

#include "psrdada_cpp/detail/double_device_buffer.cu"

#endif //PSRDADA_CPP_DOUBLE_DEVICE_BUFFER_HPP
