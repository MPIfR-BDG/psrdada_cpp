#ifndef PSRDADA_CPP_DADA_OUTPUT_STREAM_HPP
#define PSRDADA_CPP_DADA_OUTPUT_STREAM_HPP

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

    template <class HandlerType>
    DadaOutputStream<HandlerType>::DadaOutputStream(key_t key, MultiLog& log)
    : _writer(key,log)
    {
    }

    template <class HandlerType>
    DadaOutputStream<HandlerType>::~DadaOutputStream()
    {
    }

    template <class HandlerType>
    DadaWriteClient const& client() const;
    {
        return _writer;
    }

    template <class HandlerType>
    void DadaOutputStream<HandlerType>::init(RawBytes& in)
    {
        _writer.reset();
        auto& stream = _writer.header_stream();
        auto& buffer = stream.next();
        memcpy(in.ptr(),buffer.ptr(),in.used_bytes());
        stream.realease();
    }

    template <class HandlerType>
    void DadaOutputStream<HandlerType>::operator()(RawBytes& in)
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