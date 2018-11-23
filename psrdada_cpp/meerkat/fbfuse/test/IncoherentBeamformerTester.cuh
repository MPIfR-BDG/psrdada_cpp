#ifndef PSRDADA_CPP_MEERKAT_FBFUSE_INCOHERENTBEAMFORMERTESTER_CUH
#define PSRDADA_CPP_MEERKAT_FBFUSE_INCOHERENTBEAMFORMERTESTER_CUH

#include "psrdada_cpp/meerkat/fbfuse/PipelineConfig.hpp"
#include "psrdada_cpp/meerkat/fbfuse/IncoherentBeamformer.cuh"
#include "thrust/host_vector.h"
#include <gtest/gtest.h>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace test {

class IncoherentBeamformerTester: public ::testing::Test
{
public:
    typedef IncoherentBeamformer::VoltageVectorType DeviceVoltageVectorType;
    typedef thrust::host_vector<char2> HostVoltageVectorType;
    typedef IncoherentBeamformer::PowerVectorType DevicePowerVectorType;
    typedef thrust::host_vector<int8_t> HostPowerVectorType;

protected:
    void SetUp() override;
    void TearDown() override;

public:
    IncoherentBeamformerTester();
    ~IncoherentBeamformerTester();

protected:
    void beamformer_c_reference(
        HostVoltageVectorType const& taftp_voltages,
        HostPowerVectorType& tf_powers,
        int nchannels,
        int naccumulate,
        int ntimestamps,
        int nantennas,
        int npol,
        float scale,
        float offset);

    void compare_against_host(
        DeviceVoltageVectorType const& taftp_voltages_gpu,
        DevicePowerVectorType& tf_powers_gpu,
        int ntimestamps);

protected:
    PipelineConfig _config;
    cudaStream_t _stream;
};

} //namespace test
} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

#endif //PSRDADA_CPP_MEERKAT_FBFUSE_INCOHERENTBEAMFORMERTESTER_CUH
