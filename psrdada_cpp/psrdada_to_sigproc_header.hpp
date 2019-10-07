#ifndef PSRDADA_CPP_PSRDADA_TO_SIGPROC_HEADER_HPP
#define PSRDADA_CPP_PSRDADA_TO_SIGPROC_HEADER_HPP

#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/sigprocheader.hpp"

namespace psrdada_cpp {


template <class HandlerType>
class PsrDadaToSigprocHeader
{

public:
    PsrDadaToSigprocHeader(std::uint32_t beamnum, HandlerType& handler);
    ~PsrDadaToSigprocHeader();

    /**
     * @brief      A header manipulation method for PSRDADA and SIGPROC
     *
     *
     * @detail     A class that converts the PSRDADA header to a SIGPROC
     * 	           header before writing it out to the header block of the
     * 	           DADA buffer. This conversion is needed so that the down
     * 	           stream pipeline can handle the header format.
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

    /**
     * @brief      Set the SIGPROC header
     */
    void header(SigprocHeader const& header);

    /**
     * @brief      Get the SIGPROC header
     */
    SigprocHeader& header();

private:
    HandlerType& _handler;
    std::uint32_t _beamnum;
    SigprocHeader _sh;
    char* _optr;
};


} //psrdada_cpp

#include "psrdada_cpp/detail/psrdada_to_sigproc_header.cpp"
#endif //PSRDADA_CPP_PSRDADA_TO_SIGPROC_HEADER_HPP
