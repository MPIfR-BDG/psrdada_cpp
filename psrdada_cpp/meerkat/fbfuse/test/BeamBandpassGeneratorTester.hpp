#ifndef PSRDADA_CPP_MEERKAT_FBFUSE_BEAMBANDPASSGENERATORTESTER_CUH
#define PSRDADA_CPP_MEERKAT_FBFUSE_BEAMBANDPASSGENERATORTESTER_CUH

#include "psrdada_cpp/meerkat/fbfuse/BeamBandpassGenerator.hpp"
#include <gtest/gtest.h>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace test {

class BeamBandpassGeneratorTester: public ::testing::Test
{
protected:
    void SetUp() override;
    void TearDown() override;

public:
    BeamBandpassGeneratorTester();
    ~BeamBandpassGeneratorTester();
};

} //namespace test
} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

#endif //PSRDADA_CPP_MEERKAT_FBFUSE_BEAMBANDPASSGENERATORTESTER_CUH
