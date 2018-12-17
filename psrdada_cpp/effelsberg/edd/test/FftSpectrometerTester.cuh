#ifndef PSRDADA_CPP_EFFELSBERG_EDD_FFTSPECTROMETERTESTER_CUH
#define PSRDADA_CPP_EFFELSBERG_EDD_FFTSPECTROMETERTESTER_CUH

#include "psrdada_cpp/effelsberg/edd/FftSpectrometer.cuh"
#include "psrdada_cpp/dada_db.hpp"
#include "thrust/host_vector.h"
#include <gtest/gtest.h>
#include <vector>

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {
namespace test {

class FftSpectrometerTester: public ::testing::Test
{
protected:
    void SetUp() override;
    void TearDown() override;

public:
    FftSpectrometerTester();
    ~FftSpectrometerTester();

    void performance_test(
        std::size_t nchans, std::size_t tscrunch, 
        std::size_t nsamps_out, std::size_t nbits);

};

} //namespace test
} //namespace edd
} //namespace meerkat
} //namespace psrdada_cpp

#endif //PSRDADA_CPP_EFFELSBERG_EDD_FFTSPECTROMETERTESTER_CUH
