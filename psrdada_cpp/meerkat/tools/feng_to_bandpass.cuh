#ifndef PSRDADA_CPP_MEERKAT_TOOLS_FENG_TO_BANDPASS_HPP
#define PSRDADA_CPP_MEERKAT_TOOLS_FENG_TO_BANDPASS_HPP

#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/meerkat/constants.hpp"
#include "psrdada_cpp/raw_bytes.hpp"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

namespace psrdada_cpp {
namespace meerkat {
namespace tools {
namespace kernels {

    /**
     * @brief      Convert data that is in MeerKAT F-engine order into
     *             a spectrum for each antenna
     *
     * @detail     The heaps from the the MeerKAT F-engine are in in FTP
     *             order with the T = 256 and P = 2. The number of frequency
     *             channels in a heap is variable but is always a power of two.
     *             The data itself is 8-bit complex (8-bit real, 8-bit imaginary).
     *             In this kernel we perform a vectorised char2 load such that
     *             each thread gets a complete complex voltage.
     *
     *             As each block with process all heaps from a given antenna
     *             for a given frequency channel we use 512 threads per block
     *             which matches nicely with the inner TP dimensions of the heaps.
     *
     *             The heaps themselves are ordered in TAF (A=antenna) order. As
     *             such the full order of the input can be considered to be
     *             tAFFTP which simplifies to tAFTP (using small t and big T
     *             to disambiguate between the two time axes). Each block of
     *             threads will process TP for all t (for one A and one F).
     *
     * @param      in               Input buffer
     * @param      out              Output buffer
     * @param[in]  nchans           The number of frequency chans
     * @param[in]  nants            The number of antennas
     * @param[in]  ntimestamps      The number of timestamps (this corresponds to the
     *                              number of heaps in the time axis)
     */
    __global__
    void feng_heaps_to_bandpass(
        char2* __restrict__ in, float* __restrict__ out,
        int nchans, int nants,
        int ntimestamps);

}

template <class HandlerType>
class FengToBandpass
{
public:
    FengToBandpass(std::size_t nchans, std::size_t nants, HandlerType& handler);
    ~FengToBandpass();

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
    std::size_t _nchans;
    std::size_t _natnennas;
    HandlerType& _handler;
    thrust::device_vector<char2> _input;
    thrust::device_vector<float> _output;
};


} //tools
} //meerkat
} //psrdada_cpp

#include "psrdada_cpp/meerkat/tools/detail/feng_to_bandpass.cu"
#endif //PSRDADA_CPP_MEERKAT_TOOLS_FENG_TO_BANDPASS_HPP