#include "psrdada_cpp/transpose_to_data.hpp"
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

         for (j =0; j < _nsamples; j++)
            {
                for (k = 0; k < _ntime ; k++)
                {

                    for (l = 0; l < _nfreq ; l++)
                    {


                        for (m=0;m < _nchans ; m++)
                        {

                            transposed_data.ptr()[a] = block.ptr()[m + _ntime * _nchans * _nsamples * l + _nchans * (j * _ntime + k) + _nsamples * _nchans * _ntime* _nfreq * beamnum];
                            a++;

                        }


                    }


                }

         }
        _handler(block);
        return false;
    }

} //psrdada_cpp
