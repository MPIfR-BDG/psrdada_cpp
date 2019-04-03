#include "psrdada_cpp/psrdada_to_sigproc_header.hpp"
#include "psrdada_cpp/psrdadaheader.hpp"
#include "psrdada_cpp/sigprocheader.hpp"
#include "psrdada_cpp/cli_utils.hpp"
#include <thread>

namespace psrdada_cpp {

    template <class HandlerType>
    PsrDadaToSigprocHeader<HandlerType>::PsrDadaToSigprocHeader(HandlerType& handler)
    : _handler(handler)
    {
    }

    template <class HandlerType>
    PsrDadaToSigprocHeader<HandlerType>::~PsrDadaToSigprocHeader()
    {
    }

    template <class HandlerType>
    void PsrDadaToSigprocHeader<HandlerType>::init(RawBytes& block)
    {
        SigprocHeader h;
        PsrDadaHeader ph;
        ph.from_bytes(block);
        std::memset(block.ptr(), 0, block.total_bytes());
        h.write_header(block,ph);
        headersize(h.header_size());
        block.used_bytes(block.total_bytes());
        _handler.init(block);
    }

    template <class HandlerType>
    bool PsrDadaToSigprocHeader<HandlerType>::operator()(RawBytes& block)
    {
        _handler(block);
        return false;
    }

    template <class HandlerType>
    std::size_t PsrDadaToSigprocHeader<HandlerType>::headersize()
    {
        return _headersize;
    }

    template <class HandlerType>
    void PsrDadaToSigprocHeader<HandlerType>::headersize(std::size_t headersize)
    {
        _headersize = headersize;
    }
 

} //psrdada_cpp
