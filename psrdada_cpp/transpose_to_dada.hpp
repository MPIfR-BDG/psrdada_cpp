#ifndef PSRDADA_CPP_TRANSPOSE_TO_DADA_HPP
#define PSRDADA_CPP_TRANSPOSE_TO_DADA_HPP

#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/constants.hpp"


namespace psrdada_cpp {


template <class HandlerType>
class TransposeToDada
{
public:
    TransposeToDada(std::size_t beamnum, HandlerType& handler);
    ~TransposeToDada();

    /**
     * @brief      A callback to be called on connection
     *             to a ring buffer.
     *
     * @detail     The first available header block in the
     *             in the ring buffer is provided as an argument.
     *             It is here that header parameters could be read
     *             if desired.
     *
     * @param      block  A RawBytes object wrapping a DADA header buffer
     */
    void init(RawBytes& block);

    /**
     * @brief      A callback to be called on acqusition of a new
     *             data block.
     *
     * @param      block  A RawBytes object wrapping a DADA data buffer
     */
    bool operator()(RawBytes& block);

private:
    std::size_t _beamnum;
    HandlerType& _handler;
    std::uint32_t _nchans;
    std::uint32_t _nsamples;
    std::uint32_t _ntime;
    std::uint32_t _nfreq;

};


} //psrdada_cpp

#include "psrdada_cpp/detail/transpose_to_dada.cpp"
#endif //PSRDADA_CPP_TRANSPOSE_TO_DADA_HPP
