#include "psrdada_cpp/dada_output_stream.hpp"

namespace psrdada_cpp {

    DadaOutputStream::DadaOutputStream(key_t key, MultiLog& log)
    : _writer(key,log)
    {
    }

    DadaOutputStream::~DadaOutputStream()
    {
    }

    DadaWriteClient const& DadaOutputStream::client() const
    {
        return _writer;
    }

    void DadaOutputStream::init(RawBytes& in)
    {
        _writer.reset();
        auto& stream = _writer.header_stream();
        auto& buffer = stream.next();
        memcpy(in.ptr(),buffer.ptr(),in.used_bytes());
        buffer.used_bytes(in.used_bytes());
        stream.release();
    }

    void DadaOutputStream::operator()(RawBytes& in)
    {
        auto& stream = _writer.data_stream();
        auto& out = stream.next();
        memcpy(in.ptr(),out.ptr(),in.used_bytes());
        out.used_bytes(in.used_bytes());
        stream.release();
    }

} //psrdada_cpp