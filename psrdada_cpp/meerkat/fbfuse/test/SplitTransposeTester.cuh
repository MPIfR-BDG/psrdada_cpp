#ifndef PSRDADA_CPP_MEERKAT_FBFUSE_SPLITTRANSPOSETESTER_CUH
#define PSRDADA_CPP_MEERKAT_FBFUSE_SPLITTRANSPOSETESTER_CUH

#include "psrdada_cpp/meerkat/fbfuse/PipelineConfig.hpp"
#include "psrdada_cpp/meerkat/fbfuse/SplitTranspose.cuh"
#include "thrust/host_vector.h"
#include <gtest/gtest.h>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace test {

class SplitTransposeTester: public ::testing::Test
{
public:
    typedef SplitTranspose::VoltageType DeviceVoltageType;
    typedef thrust::host_vector<char2> HostVoltageType;


protected:
    void SetUp() override;
    void TearDown() override;

public:
    SplitTransposeTester();
    ~SplitTransposeTester();

protected:
    void transpose_c_reference(
        HostVoltageType const& input,
        HostVoltageType& output,
        int total_nantennas,
        int used_nantennas,
        int start_antenna,
        int nchans,
        int ntimestamps);

    void compare_against_host(
        VoltageType const& gpu_input,
        VoltageType const& gpu_output,
        std::size_t ntimestamps)

protected:
    PipelineConfig _config;
    cudaStream_t _stream;
};

} //namespace test
} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

#endif //PSRDADA_CPP_MEERKAT_FBFUSE_SPLITTRANSPOSETESTER_CUH
