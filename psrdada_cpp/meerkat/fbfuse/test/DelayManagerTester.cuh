#ifndef PSRDADA_CPP_MEERKAT_FBFUSE_DELAYMANAGERTEST_CUH
#define PSRDADA_CPP_MEERKAT_FBFUSE_DELAYMANAGERTEST_CUH

#include "psrdada_cpp/meerkat/fbfuse/PipelineConfig.hpp"
#include "psrdada_cpp/meerkat/fbfuse/DelayManager.cuh"
#include <gtest/gtest.h>

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
    void compare_against_host(DelayManager::DelayVectorType const&, DelayModel*);

protected:
    PipelineConfig _config;
    cudaStream_t _stream;
};

} //namespace test
} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

#endif //PSRDADA_CPP_MEERKAT_FBFUSE_DELAYMANAGERTEST_CUH
