#ifndef PSRDADA_CPP_RAW_BYTES_HPP
#define PSRDADA_CPP_RAW_BYTES_HPP

#include "psrdada_cpp/common.hpp"

namespace psrdada_cpp {

    class RawBytes
    {
    private:
        char* _ptr;
        std::size_t _total_bytes;
        std::size_t _used_bytes;

    public:
        RawBytes(char* ptr, std::size_t total, std::size_t used=0);
        RawBytes(RawBytes const&) = delete;
        ~RawBytes();
        std::size_t total_bytes();
        std::size_t used_bytes();
        void used_bytes(std::size_t);
        char* ptr();
    };

} //namespace psrdada_cpp
#endif //PSRDADA_CPP_RAW_BYTES_HPP