#ifndef PSRDADA_CPP_MEERKAT_FBFUSE_PIPELINETESTER_CUH
#define PSRDADA_CPP_MEERKAT_FBFUSE_PIPELINETESTER_CUH

#include "psrdada_cpp/meerkat/fbfuse/PipelineConfig.hpp"
#include "psrdada_cpp/meerkat/fbfuse/Pipeline.cuh"
#include "thrust/host_vector.h"
#include <gtest/gtest.h>
#include <memory>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace test {

class PipelineTester: public ::testing::Test
{
public:


protected:
    void SetUp() override;
    void TearDown() override;

public:
    PipelineTester();
    ~PipelineTester();

protected:
    PipelineConfig _config;
};

} //namespace test
} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

#endif //PSRDADA_CPP_MEERKAT_FBFUSE_PIPELINETESTER_CUH
