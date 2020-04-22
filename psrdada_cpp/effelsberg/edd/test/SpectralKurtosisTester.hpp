#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/effelsberg/edd/SKRfiReplacement.hpp"
#include "psrdada_cpp/effelsberg/edd/SpectralKurtosis.hpp"
#include "psrdada_cpp/effelsberg/edd/test/SKTestVector.hpp"
#include <gtest/gtest.h>

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {
namespace test {

class SpectralKurtosisTester: public ::testing::Test
{
public:
    SpectralKurtosisTester();
    ~SpectralKurtosisTester();
    /**
     * @brief        creates SKTestVector class instance and generates test vector
     *
     */
    void test_vector_generation(std::size_t sample_size, std::size_t window_size, bool with_rfi,
                                float rfi_freq, float rfi_amp, const std::vector<int> &rfi_window_indices,
				std::vector<std::complex<float>> &samples);
    /**
     * @brief       creates SpectralKurtosis class instance and computes spectral kurtosis
     *
     */
    void sk_computation(std::size_t nch,std::size_t window_size, 
                        const std::vector<std::complex<float>> &samples,
			RFIStatistics &stat);
protected:
    void SetUp() override;
    void TearDown() override;
};

} //test
} //edd
} //effelsberg
} //psrdada_cpp




