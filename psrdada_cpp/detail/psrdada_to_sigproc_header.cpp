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
        header(h);
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
    void PsrDadaToSigprocHeader<HandlerType>::header(SigprocHeader const& h)
    {
        _sh = h;
    }

    template <class HandlerType>
    SigprocHeader& PsrDadaToSigprocHeader<HandlerType>::header()
    {
        return _sh;
    }
 

} //psrdada_cpp
