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
	char* new_ptr = new char[block.total_bytes()];
	RawBytes new_block(new_ptr,block.total_bytes(),std::size_t(0));
        h.write_header(new_block,ph);
	new_block.used_bytes(new_block.total_bytes());
        _handler.init(new_block);
    }

    template <class HandlerType>
    bool PsrDadaToSigprocHeader<HandlerType>::operator()(RawBytes& block)
    {
	_handler(block);
        return false;
    }
 

} //psrdada_cpp
