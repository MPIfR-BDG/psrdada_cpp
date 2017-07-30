#include "psrdada_cpp/dada_junkdb.hpp"

namespace psrdada_cpp
{
    JunkDb::JunkDb(key_t key, MultiLog& log, std::size_t nbytes)
    : DadaIoLoopWriter<JunkDb>(key, log)
    , _nbytes(nbytes)
    , _infinite(_nbytes==0)
    {
    }

    JunkDb::~JunkDb()
    {
    }

    void JunkDb::on_connect(RawBytes& block)
    {
        std::fill(block.ptr(), block.ptr()+block.total_bytes(),1);
    }

    bool JunkDb::on_next(RawBytes& block)
    {
        if (_infinite)
        {
            std::fill(block.ptr(), block.ptr()+block.total_bytes(),0);
            block.used_bytes(block.total_bytes());
            return false;
        }
        else
        {
           std::size_t bytes_to_write = std::min(_nbytes,block.total_bytes());
           std::fill(block.ptr(), block.ptr()+bytes_to_write,0);
           block.used_bytes(bytes_to_write);
           _nbytes -= bytes_to_write;
           return _nbytes <= 0;
        }
    }
} //namespace psrdada_cpp