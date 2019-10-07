#ifndef PSRDADA_CPP_MEERKAT_APSUSE_TEST_BEAMCAPTURECONTROLLERTESTER_HPP
#define PSRDADA_CPP_MEERKAT_APSUSE_TEST_BEAMCAPTURECONTROLLERTESTER_HPP

#include "psrdada_cpp/meerkat/apsuse/BeamCaptureController.hpp"
#include <gtest/gtest.h>

namespace psrdada_cpp {
namespace meerkat {
namespace apsuse {
namespace test {

class BeamCaptureControllerTester: public ::testing::Test
{
protected:
    void SetUp() override;
    void TearDown() override;

public:
    BeamCaptureControllerTester();
    ~BeamCaptureControllerTester();

};

} //namespace test
} //namespace apsuse
} //namespace meerkat
} //namespace psrdada_cpp

#endif //PSRDADA_CPP_MEERKAT_APSUSE_TEST_BEAMCAPTURECONTROLLERTESTER_HPP
