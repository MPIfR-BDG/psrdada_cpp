#ifndef PSRDADA_CPP_MEERKAT_TOOLS_FENG_TO_DADA_HPP
#define PSRDADA_CPP_MEERKAT_TOOLS_FENG_TO_DADA_HPP

#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/dada_io_loop_reader.hpp"
#include "psrdada_cpp/meerkat/constants.hpp"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

namespace psrdada_cpp {
namespace meerkat {
namespace tools {
namespace kernels {

    __global__
    void feng_heaps_to_dada(
        char2* __restrict__ in, float* __restrict__ out,
        int nchans, int ntimestamps);

}

class FengToDada:
    public DadaIoLoopReader<FengToDada>
{
public:
    FengToDada(key_t key, MultiLog& log, std::size_t nchans);
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
    void on_connect(RawBytes& block);

    /**
     * @brief      A callback to be called on acqusition of a new
     *             data block.
     *
     * @param      block  A RawBytes object wrapping a DADA data buffer
     */
    void on_next(RawBytes& block);

private:
    void write_output_file();
    std::size_t _nchans;
    std::size_t _dump_counter;
    thrust::device_vector<char2> _input;
    thrust::device_vector<float> _output;
    thrust::host_vector<float> _h_output;

};


} //tools
} //meerkat
} //psrdada_cpp

#endif //PSRDADA_CPP_MEERKAT_TOOLS_FENG_TO_DADA_HPP