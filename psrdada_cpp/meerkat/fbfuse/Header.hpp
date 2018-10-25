#ifndef PSRDADA_CPP_MEERKAT_FBFUSE_HEADER_HPP
#define PSRDADA_CPP_MEERKAT_FBFUSE_HEADER_HPP

#include "psrdada_cpp/common.hpp"

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {

class Header
{
public:
    explicit Header(RawBlock&);
    ~Header();
    Header(Header const&) = delete;

    template <typename T>
    T get(char const* key);

    template <typename T>
    void set(char const* key, T value);

    void purge();

private:
    void fetch_header_string(char const* key);

private:
    RawBlock& _header;
    char _buffer[1024];
};

} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

#include "psrdada_cpp/meerkat/fbfuse/detail/Header.cpp"

#endif //PSRDADA_CPP_MEERKAT_FBFUSE_HEADER_HPP
