#include "psrdada_cpp/meerkat/fbfuse/test/DelayManagerTester.cuh"
#include "psrdada_cpp/meerkat/fbfuse/DelayEngineSimulator.cuh"
#include "psrdada_cpp/meerkat/fbfuse/fbfuse_constants.hpp"
#include "psrdada_cpp/cuda_utils.hpp"
#include "thrust/host_vector.h"

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace test {

DelayManagerTester::DelayManagerTester()
    : ::testing::Test()
    , _stream(0)
{
    _config.delay_buffer_shm("test_delay_buffer_shm");
    _config.delay_buffer_sem("test_delay_buffer_sem");
    _config.delay_buffer_mutex("test_delay_buffer_mutex");
}

DelayManagerTester::~DelayManagerTester()
{
}

void DelayManagerTester::SetUp()
{
    CUDA_ERROR_CHECK(cudaStreamCreate(&_stream));
}

void DelayManagerTester::TearDown()
{
    CUDA_ERROR_CHECK(cudaStreamDestroy(_stream));
}

void DelayManagerTester::compare_against_host(DelayManager::DelayVectorType const& delays)
{
    // Implicit sync copy back to host
    thrust::host_vector<DelayManager::DelayType> host_delays = delays;
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    for (int ii=0; ii < FBFUSE_CB_NBEAMS * FBFUSE_CB_NANTENNAS; ++ii)
    {
        ASSERT_EQ(_delay_model->delays[ii].x, host_delays[ii].x);
        ASSERT_EQ(_delay_model->delays[ii].y, host_delays[ii].y);
    }
}

TEST_F(DelayManagerTester, test_updates)
{
    DelayEngineSimulator simulator(_config);
    DelayManager delay_manager(_config, _stream);
    simulator.update_delays();
    auto const& delay_vector = delay_manager.delays();
    compare_against_host(delay_vector);
    std::memset(static_cast<void*>(_delay_model->delays), 1, sizeof(_delay_model->delays));
    simulator.update_delays();
    auto const& delay_vector_2 = delay_manager.delays();
    compare_against_host(delay_vector_2);
}

TEST_F(DelayManagerTester, test_bad_keys)
{
    DelayEngineSimulator simulator(_config);
    _config.delay_buffer_shm("bad_test_delay_buffer_shm");
    _config.delay_buffer_sem("bda_test_delay_buffer_sem");
    _config.delay_buffer_mutex("bad_test_delay_buffer_mutex");
    ASSERT_ANY_THROW(DelayManager(config, _stream));
}

} //namespace test
} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

