#include "psrdada_cpp/dada_null_sink.hpp"

namespace psrdada_cpp
{
    NullSink::NullSink()
    {
    }

    NullSink::~NullSink()
    {
    }

    void NullSink::init(RawBytes& /*block*/)
    {
    }

    bool NullSink::operator()(RawBytes& /*block*/)
    {
        return false;
    }
} //namespace psrdada_cpp