#ifndef PSRDADA_CPP_DADA_DISK_SINK_HPP
#define PSRDADA_CPP_DADA_DISK_SINK_HPP

#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/common.hpp"
#include <fstream>

namespace psrdada_cpp
{
    class DiskSink
    {
    public:
        DiskSink(std::string prefix);
        ~DiskSink();
        void init(RawBytes&);
        bool operator()(RawBytes&);

    public:
        std::string _prefix;
        std::size_t _counter;
        std::ofstream _current_file;
    };
} //namespace psrdada_cpp
#endif //PSRDADA_CPP_DADA_DISK_SINK_HPP