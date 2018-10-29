#ifndef PSRDADA_CPP_MEERKAT_FBFUSE_DELAYMANAGERTEST_CUH
#define PSRDADA_CPP_MEERKAT_FBFUSE_DELAYMANAGERTEST_CUH

#include "psrdada_cpp/meerkat/fbfuse/PipelineConfig.hpp"
#include "psrdada_cpp/meerkat/fbfuse/DelayManager.cuh"
#include <gtest/gtest.h>
#include <semaphore.h>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace test {

class DelayManagerTester: public ::testing::Test
{
protected:
    void SetUp() override;
    void TearDown() override;

public:
    DelayManagerTester();
    ~DelayManagerTester();

protected:
    PipelineConfig _config;
    int _shm_fd;
    void* _shm_ptr;
    sem_t* _sem_id;
    sem_t* _mutex_id;
    DelayModel* _delay_model;
    cudaStream_t _stream;
};

} //namespace test
} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

#endif //PSRDADA_CPP_MEERKAT_FBFUSE_DELAYMANAGERTEST_CUH
