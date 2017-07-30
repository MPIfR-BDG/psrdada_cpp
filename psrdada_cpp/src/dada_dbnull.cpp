#include "psrdada_cpp/dada_dbnull.hpp"

namespace psrdada_cpp
{
    DbNull::DbNull(key_t key, MultiLog& log, std::size_t nbytes)
    : DadaIoLoopReader<DbNull>(key, log)
    , _nbytes(nbytes)
    , _infinite(_nbytes==0)
    {
    }

    DbNull::~DbNull()
    {
    }

    void DbNull::on_connect(RawBytes& block)
    {
    }

    void DbNull::on_next(RawBytes& block)
    {
        if (!_infinite)
        {
            std::size_t bytes_to_read = std::min(_nbytes,block.used_bytes());
            _nbytes -= bytes_to_read;
            if (_nbytes <= 0)
                stop();
        }
    }
} //namespace psrdada_cpp