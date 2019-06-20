#ifndef PSRDADA_CPP_MEERKAT_FBFUSE_WEIGHTSMANAGERTESTER_CUH
#define PSRDADA_CPP_MEERKAT_FBFUSE_WEIGHTSMANAGERTESTER_CUH

#include "psrdada_cpp/meerkat/fbfuse/PipelineConfig.hpp"
#include "psrdada_cpp/meerkat/fbfuse/WeightsManager.cuh"
#include <gtest/gtest.h>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace test {

class WeightsManagerTester: public ::testing::Test
{
public:
    typedef WeightsManager::DelayVectorType DelayVectorType;
    typedef WeightsManager::WeightsVectorType WeightsVectorType;
    typedef WeightsManager::TimeType TimeType;

protected:
    void SetUp() override;
    void TearDown() override;

public:
    WeightsManagerTester();
    ~WeightsManagerTester();

protected:
    void calc_weights_c_reference(
        thrust::host_vector<float2> const& delay_models,
        thrust::host_vector<char2>& weights,
        std::vector<double> const& channel_frequencies,
        int nantennas,
        int nbeams,
        int nchans,
        double current_epoch,
        double delay_epoch,
        double tstep,
        int ntsteps);

    void compare_against_host(DelayVectorType const& delays,
        WeightsVectorType const& weights,
        TimeType current_epoch, TimeType delay_epoch);

protected:
    PipelineConfig _config;
    cudaStream_t _stream;
};

} //namespace test
} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

#endif //PSRDADA_CPP_MEERKAT_FBFUSE_WEIGHTSMANAGERTESTER_CUH
