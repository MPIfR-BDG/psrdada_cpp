#ifndef PSRDADA_CPP_EFFELSBERG_EDD_DETECTORACCUMULATORTESTER_CUH
#define PSRDADA_CPP_EFFELSBERG_EDD_DETECTORACCUMULATORTESTER_CUH

#include "psrdada_cpp/effelsberg/edd/DetectorAccumulator.cuh"
#include "thrust/host_vector.h"
#include <gtest/gtest.h>
#include <vector>

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {
namespace test {

class DetectorAccumulatorTester: public ::testing::Test
{
public:
    typedef thrust::host_vector<float2> InputType;
    typedef thrust::host_vector<int8_t> OutputType;

protected:
    void SetUp() override;
    void TearDown() override;

public:
    DetectorAccumulatorTester();
    ~DetectorAccumulatorTester();

protected:
    void detect_c_reference(
        InputType const& input,
        OutputType& output,
        int nchans,
        int tscrunch,
        float scale,
        float offset);

    void compare_against_host(
        DetectorAccumulator::OutputType const& gpu_output,
        OutputType const& host_output);

protected:
    cudaStream_t _stream;
};

} //namespace test
} //namespace edd
} //namespace meerkat
} //namespace psrdada_cpp

#endif //PSRDADA_CPP_EFFELSBERG_EDD_DETECTORACCUMULATORTESTER_CUH
