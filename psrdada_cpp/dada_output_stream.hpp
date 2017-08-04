#ifndef PSRDADA_CPP_DADA_OUTPUT_STREAM_HPP
#define PSRDADA_CPP_DADA_OUTPUT_STREAM_HPP

#include "psrdada_cpp/dada_write_client.hpp"
#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/common.hpp"

namespace psrdada_cpp
{
    class DadaOutputStream
    {
    public:
        DadaOutputStream(key_t key, MultiLog& log);
        ~DadaOutputStream();
        void init();
        void operator()();
        DadaWriteClient const& client() const;

    private:
        DadaWriteClient _writer;
    };

    DadaOutputStream::DadaOutputStream(key_t key, MultiLog& log)
    : _writer(key,log)
    {
    }

    DadaOutputStream::~DadaOutputStream()
    {
    }

    DadaWriteClient const& DadaOutputStream::client() const;
    {
        return _writer;
    }

    void DadaOutputStream::init(RawBytes& in)
    {
        _writer.reset();
        auto& stream = _writer.header_stream();
        auto& buffer = stream.next();
        memcpy(in.ptr(),buffer.ptr(),in.used_bytes());
        stream.realease();
    }

    void DadaOutputStream::operator()(RawBytes& in)
    {
        if (!_writer)
        {
            throw std::runtime_error("operator() called without call to init() first");
        }
        auto& stream = _writer.data_stream();
        auto& out = stream.next();
        memcpy(in.ptr(),out.ptr(),in.used_bytes());
        stream.release();
    }


}

#endif //PSRDADA_CPP_DADA_OUTPUT_STREAM_HPP