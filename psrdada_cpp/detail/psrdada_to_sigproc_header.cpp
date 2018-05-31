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
    	/* Do the psrdada to sigproc conversion here */
        SigprocHeader h;
        PsrDadaHeader ph;
        ph.from_bytes(block);
        //Not sure if overwriting works--KR
        h.write_header(block,ph);
        _handler.init(block);
    }

    template <class HandlerType>
    bool PsrDadaToSigprocHeader<HandlerType>::operator()(RawBytes& block)
    {
    	_handler(block);
        return false;
    }
 

} //psrdada_cpp
