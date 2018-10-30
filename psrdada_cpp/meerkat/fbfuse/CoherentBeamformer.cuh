#ifndef PSRDADA_CPP_MEERKAT_FBFUSE_COHERENTBEAMFORMER_HPP
#define PSRDADA_CPP_MEERKAT_FBFUSE_COHERENTBEAMFORMER_HPP

#include "psrdada_cpp/meerkat/fbfuse/PipelineConfig.hpp"
#include "thrust/device_vector.h"
#include "cuda.h"

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace kernels {

struct char4x2
{
    char4 x;
    char4 y;
};

struct char2x4
{
    char2 x;
    char2 y;
    char2 z;
    char2 w;
};

__forceinline__ __device__
void dp4a(int &c, const int &a, const int &b);

__forceinline__ __device__
int2 int2_transpose(int2 const &input);

__global__
void bf_aptf_general_k(
    int2 const* __restrict__ aptf_voltages,
    int2 const* __restrict__ apbf_weights,
    int8_t* __restrict__ ftb_powers,
    float output_scale,
    float output_offset,
    int nsamples);

} //namespace kernels


class CoherentBeamformer
{
public:
    // FTPA order
    typedef thrust::device_vector<char2> VoltageVectorType;
    // TBTF order
    typedef thrust::device_vector<char> PowerVectorType;
    // order??
    typedef thrust::device_vector<char2> WeightsVectorType;

public:
    CoherentBeamformer(PipelineConfig const&);
    ~CoherentBeamformer();
    CoherentBeamformer(CoherentBeamformer const&) = delete;
    void beamform(VoltageVectorType const& input,
        WeightsVectorType const& weights,
        PowerVectorType& output,
        cudaStream_t stream);

private:
    PipelineConfig const& _config;
    std::size_t _size_per_sample;
    std::size_t _expected_weights_size;
};

} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

#endif //PSRDADA_CPP_MEERKAT_FBFUSE_COHERENTBEAMFORMER_HPP
