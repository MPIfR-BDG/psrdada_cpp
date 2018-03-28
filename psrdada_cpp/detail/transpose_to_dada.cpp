#include "psrdada_cpp/transpose_to_dada.hpp"
#include "psrdada_cpp/cli_utils.hpp"

namespace psrdada_cpp {

    template <class HandlerType>
    TransposeToDada<HandlerType>::TransposeToDada(std::size_t beamnum, HandlerType& handler)
    : _beamnum(beamnum)
    , _handler(handler
    , _nchans(128)
    , _nsamples(64)
    , _ntime(64)
    , _nfreq(32)
    {
    }

    template <class HandlerType>
    TransposeToDada<HandlerType>::~TransposeToDada()
    {
    }

    template <class HandlerType>
    void TransposeToDada<HandlerType>::init(RawBytes& block)
    {
        _handler.init(block);
    }

    template <class HandlerType>
    bool TransposeToDada<HandlerType>::operator()(RawBytes& block)
    {
      
        char o_data[_nchans*_nsamples*_ntime*_nfreq];
        RawBytes transpose(o_data,std::size_t(_nchans*_nsamples*_ntime*_nfreq),std::size_t(0));
        transpose::do_transpose(block,transpose,_beamnum);
        _handler(block);
        return false;
    }

} //psrdada_cpp
