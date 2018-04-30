#include "psrdada_cpp/raw_bytes.hpp"

namespace psrdada_cpp {

    RawBytes::RawBytes(char* ptr, std::size_t total, std::size_t used)
    : _ptr(ptr)
    , _total_bytes(total)
    , _used_bytes(used)
    {
    }

    RawBytes::~RawBytes()
    {
    }

    std::size_t RawBytes::total_bytes() const
    {
        return _total_bytes;
    }

    std::size_t RawBytes::used_bytes() const
    {
        return _used_bytes;
    }

    void RawBytes::used_bytes(std::size_t used)
    {
        _used_bytes = used;
    }

    char* RawBytes::ptr()
    {
        return _ptr;
    }

    std::size_t RawBytes::remaining_bytes() const
    {
        return _total_bytes - _used_bytes;
    }

} //namespace psrdada_cpp