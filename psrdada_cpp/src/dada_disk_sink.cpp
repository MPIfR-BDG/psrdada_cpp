#include "psrdada_cpp/dada_disk_sink.hpp"

namespace psrdada_cpp
{
    DiskSink::DiskSink(std::string prefix)
    : _prefix(prefix)
    , _counter(0)
    {
    }

    DiskSink::~DiskSink()
    {
    }

    void DiskSink::init(RawBytes& block)
    {
        if (_current_file.is_open())
        {
            _current_file.close();
        }

        std::stringstream fname;
        fname << _prefix << "_" << _counter << ".dada";
        _current_file.open(fname.str().c_str(), std::ios::out | std::ios::app | std::ios::binary);
        _current_file.write((char*) block.ptr(), block.used_bytes());
    }

    bool DiskSink::operator()(RawBytes& block)
    {
        _current_file.write((char*) block.ptr(), block.used_bytes());
        return false;
    }
} //namespace psrdada_cpp