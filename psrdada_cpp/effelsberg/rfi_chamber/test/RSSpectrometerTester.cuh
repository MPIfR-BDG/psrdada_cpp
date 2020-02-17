#ifndef PSRDADA_CPP_EFFELSBERG_RFI_CHAMBER_RSSPECTROMETERTESTER_CUH
#define PSRDADA_CPP_EFFELSBERG_RFI_CHAMBER_RSSPECTROMETERTESTER_CUH

#include <gtest/gtest.h>

namespace psrdada_cpp {
namespace effelsberg {
namespace rfi_chamber {
namespace test {

class RSSpectrometerTester: public ::testing::Test
{
protected:
    void SetUp() override;
    void TearDown() override;

public:
    RSSpectrometerTester();
    ~RSSpectrometerTester();

    void run_dc_power_test(std::size_t input_nchans, std::size_t fft_length, std::size_t naccumulate);
};

} //namespace test
} //namespace rfi_chamber
} //namespace effeslberg
} //namespace psrdada_cpp

#endif //PSRDADA_CPP_EFFELSBERG_RFI_CHAMBER_RSSPECTROMETERTESTER_CUH
