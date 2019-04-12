#ifndef PSRDADA_CPP_DADA_JUNK_SOURCE_HPP
#define PSRDADA_CPP_DADA_JUNK_SOURCE_HPP

#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/common.hpp"
#include <vector>
#include <ifstream>

namespace psrdada_cpp
{
    template <class Handler>
    void junk_source(Handler& handler,
        std::size_t header_size,
        std::string const& header_fname,
        std::size_t nbytes_per_write,
        std::size_t total_bytes)
    {
        std::vector<char> junk_header(header_size);
        std::vector<char> junk_block(nbytes_per_write);
        RawBytes header(junk_header.data(), header_size, header_size, false);
        if (header_fname)
        {
            std::ifstream headerfile (header_fname);
            if (!headerfile.is_open())
            {
                throw std::runtime_error("Unable to open header file")
            }
            headerfile.read(header.ptr(), header.total_bytes());
            headerfile.close();
        }

        handler.init(header);
        std::size_t bytes_written = 0;
        while (bytes_written < total_bytes)
        {
            RawBytes data(junk_block.data(), nbytes_per_write, nbytes_per_write, false);
            handler(data);
            bytes_written += data.used_bytes();
        }
    }
} //namespace psrdada_cpp
#endif //PSRDADA_CPP_DADA_JUNK_SOURCE_HPP