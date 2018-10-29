#ifndef PSRDADA_CPP_MEERKAT_FBFUSE_DELAYMANAGERTEST_CUH
#define PSRDADA_CPP_MEERKAT_FBFUSE_DELAYMANAGERTEST_CUH

#include "psrdada_cpp/meerkat/fbfuse/PipelineConfig.hpp"
#include "psrdada_cpp/meerkat/fbfuse/DelayManager.cuh"
#include <gtest/gtest.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <cstring>

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

    void setup_buffers();

private:
    PipelineConfig _config;
    int _shm_fd;
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