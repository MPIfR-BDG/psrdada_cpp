#ifndef PSRDADA_CPP_DADA_NULL_SINK_HPP
#define PSRDADA_CPP_DADA_NULL_SINK_HPP

#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/common.hpp"

namespace psrdada_cpp
{
    class NullSink
    {
    public:
        NullSink();
        ~NullSink();
        void init(RawBytes&);
        bool operator()(RawBytes&);
    };
} //namespace psrdada_cpp
#endif //PSRDADA_CPP_DADA_NULL_SINK_HPP