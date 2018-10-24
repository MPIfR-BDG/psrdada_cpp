#ifndef PSRDADA_CPP_DOUBLE_DEVICE_BUFFER_HPP
#define PSRDADA_CPP_DOUBLE_DEVICE_BUFFER_HPP

#include "thrust/device_vector.h"

namespace psrdada_cpp {

    template <typename T>
    class DoubleBuffer
    {
    public:
        DoubleBuffer();
        ~DoubleBuffer();
        void resize(std::size_t size);
        void swap();
        T* a() const;
        T* b() const;

    private:
        thrust::device_vector<T> _buf0;
        thrust::device_vector<T> _buf1;
        T* _a_ptr;
        T* _b_ptr;
    };

} //namespace psrdada_cpp

#include "psrdada_cpp/detail/double_device_buffer.cu"

#endif //PSRDADA_CPP_DOUBLE_DEVICE_BUFFER_HPP