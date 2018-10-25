
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

} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp
