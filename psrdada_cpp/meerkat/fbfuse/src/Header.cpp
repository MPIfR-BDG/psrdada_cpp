#include "psrdada_cpp/meerkat/fbfuse/Header.hpp"
#include "ascii_header.h"
#include <cstring>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {

Header::Header(RawBytes& header)
    : _header(header)
{

}

Header::~Header()
{

}

void Header::purge()
{
    std::memset(static_cast<void*>(_header.ptr()), 0, _header.total_bytes());
}

void Header::fetch_header_string(char const* key)
{

    if (ascii_header_get(_header.ptr(), key, "%s", _buffer) == -1)
    {
        throw std::runtime_error(
            std::string("Could not find ") + key
            + " key in DADA header.");
    }
    if (strcmp("unset", _buffer) == 0)
    {
        throw std::runtime_error(std::string("The header key ") + key + " was unset");
    }
}

template <>
long double Header::get<long double>(char const* key)
{
    fetch_header_string(key);
    long double value = std::strtold(_buffer, NULL);
    BOOST_LOG_TRIVIAL(info) << key << " = " << value;
    return value;
}

template <>
std::size_t Header::get<std::size_t>(char const* key)
{
  fetch_header_string(key);
    std::size_t value = std::strtoul(_buffer, NULL, 0);
    BOOST_LOG_TRIVIAL(info) << key << " = " << value;
    return value;
}

template <>
void Header::set<long double>(char const* key, long double value)
{
    ascii_header_set(this->_header.ptr(), key, "%ld", value);
}

template <>
void Header::set<std::size_t>(char const* key, std::size_t value)
{
    ascii_header_set(this->_header.ptr(), key, "%ul", value);
}

} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp
