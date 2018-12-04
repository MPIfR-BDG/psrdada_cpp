#include "psrdada_cpp/raw_bytes.hpp"

namespace psrdada_cpp {

    RawBytes::RawBytes(char* ptr, std::size_t total, std::size_t used, bool device_memory)
    : _ptr(ptr)
    , _total_bytes(total)
    , _used_bytes(used)
    , _on_device(device_memory)
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

    bool RawBytes::on_device() const
    {
        return _on_device;
    }

} //namespace psrdada_cpp
