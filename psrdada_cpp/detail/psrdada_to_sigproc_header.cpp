#include "psrdada_cpp/psrdada_to_sigproc_header.hpp"
#include "psrdada_cpp/psrdadaheader.hpp"
#include "psrdada_cpp/sigprocheader.hpp"
#include "psrdada_cpp/cli_utils.hpp"
#include <thread>

namespace psrdada_cpp {

    template <class HandlerType>
    PsrDadaToSigprocHeader<HandlerType>::PsrDadaToSigprocHeader(std::uint32_t beamnum, HandlerType& handler)
    : _handler(handler),
      _beamnum(beamnum)
    {
        _optr = new char[4096];
    }
    template <class HandlerType>
    PsrDadaToSigprocHeader<HandlerType>::~PsrDadaToSigprocHeader()
    {
    }

    template <class HandlerType>
    void PsrDadaToSigprocHeader<HandlerType>::init(RawBytes& block)
    {
        SigprocHeader& h = header();
        PsrDadaHeader ph;
        RawBytes outblock(_optr,block.total_bytes(),0, false);
        ph.from_bytes(block, _beamnum);
        h.write_header(outblock,ph);
        outblock.used_bytes(outblock.total_bytes());
        _handler.init(outblock);
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
