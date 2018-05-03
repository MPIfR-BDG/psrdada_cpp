#ifndef PSRDADA_CPP_DOUBLE_BUFFER_HPP
#define PSRDADA_CPP_DOUBLE_BUFFER_HPP

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
        T _buf0;
        T _buf1;
        T* _a_ptr;
        T* _b_ptr;
    };

} //namespace psrdada_cpp

#endif //PSRDADA_CPP_DOUBLE_BUFFER_HPP