#ifndef PSRDADA_CPP_MEERKAT_TOOLS_FENG_TO_DADA_HPP
#define PSRDADA_CPP_MEERKAT_TOOLS_FENG_TO_DADA_HPP

#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/meerkat/constants.hpp"
#include "thrust/device_vector.h"


namespace psrdada_cpp {
namespace meerkat {
namespace tools {
namespace kernels {

    __global__
    void feng_heaps_to_dada(
        int* __restrict__ in,
        int* __restrict__ out,
        int nchans);

}

template <class HandlerType>
class FengToDada
{
public:
    FengToDada(std::size_t nchans, HandlerType& handler);
    ~FengToDada();

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
    HandlerType& _handler;
    thrust::device_vector<int> _input;
    thrust::device_vector<int> _output;

};


} //tools
} //meerkat
} //psrdada_cpp

#include "psrdada_cpp/meerkat/tools/detail/feng_to_dada.cu"
#endif //PSRDADA_CPP_MEERKAT_TOOLS_FENG_TO_DADA_HPP
