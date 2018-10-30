#ifndef PSRDADA_CPP_MEERKAT_FBFUSE_SPLITTRANSPOSE_HPP
#define PSRDADA_CPP_MEERKAT_FBFUSE_SPLITTRANSPOSE_HPP

#include "psrdada_cpp/meerkat/fbfuse/PipelineConfig.hpp"
#include "psrdada_cpp/common.hpp"
#include "thrust/device_vector.h"

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace kernel {

__global__
void split_transpose_k(
    char2 const * __restrict__ input,
    char2 * __restrict__ output,
    int total_nantennas,
    int used_nantennas,
    int start_antenna,
    int nchans,
    int ntimestamps);

} //namespace kernel

class SplitTranspose
{
public:
    typedef thrust::device_vector<char2> VoltageType;

public:
    explicit SplitTranspose(PipelineConfig const&);
    ~SplitTranspose();
    SplitTranspose(SplitTranspose const&) = delete;
    void transpose(VoltageType const& taftp_voltages,
        VoltageType& ftpa_voltages, cudaStream_t stream);

private:
    PipelineConfig const& _config;
    std::size_t _heap_group_size;
    std::size_t _output_size_per_heap_group;

};

} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

#endif //PSRDADA_CPP_MEERKAT_FBFUSE_SPLITTRANSPOSE_HPP