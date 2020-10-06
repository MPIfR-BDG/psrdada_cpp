#include "psrdada_cpp/effelsberg/edd/test/SpectralKurtosisTester.hpp"

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {
namespace test {

SpectralKurtosisTester:: SpectralKurtosisTester():
       	::testing::Test()
{
}

SpectralKurtosisTester::~SpectralKurtosisTester()
{
}

void SpectralKurtosisTester::SetUp()
{
}

void SpectralKurtosisTester::TearDown()
{
}

void SpectralKurtosisTester::test_vector_generation(std::size_t sample_size, std::size_t window_size, 
		                                    bool with_rfi, float rfi_freq, float rfi_amp,
						    const std::vector<int> &rfi_window_indices,
						    std::vector<std::complex<float>> &samples)
{
    float m = 5;
    float std = 1;
    SKTestVector tv(sample_size, window_size, with_rfi, rfi_freq, rfi_amp, m, std);
    tv.generate_test_vector(rfi_window_indices, samples);
}

void SpectralKurtosisTester::sk_computation(std::size_t nch, std::size_t window_size, 
                                           const std::vector<std::complex<float>> &samples,
					   RFIStatistics &stat)
{
    float sk_min = 0.8;
    float sk_max = 1.2;    
    SpectralKurtosis sk(nch, window_size, sk_min, sk_max);
    sk.compute_sk(samples, stat);
}

TEST_F(SpectralKurtosisTester, sk_window_size_check)
{
    std::vector<int> rfi_window_indices{};
    std::vector<std::complex<float>> samples;
    std::size_t sample_size = 4000;
    std::size_t window_size = 150;
    test_vector_generation(sample_size, window_size, 0, 0, 0, rfi_window_indices, samples);

    RFIStatistics stat;
    std::size_t nch = 1;
    EXPECT_THROW(sk_computation(nch, window_size, samples, stat), std::runtime_error);
}

TEST_F(SpectralKurtosisTester, sk_withoutRFI)
{
    std::vector<int> rfi_window_indices{};
    std::vector<std::complex<float>> samples;
    std::size_t sample_size = 4000;
    std::size_t window_size = 400;
    test_vector_generation(sample_size, window_size, 0, 0, 0, rfi_window_indices, samples);

    RFIStatistics stat;
    std::size_t nch = 1;
    sk_computation(nch, window_size, samples, stat);
    float expected_rfi_fraction = 0;
    EXPECT_EQ(expected_rfi_fraction, stat.rfi_fraction);
}

TEST_F(SpectralKurtosisTester, sk_withRFI)
{
    std::vector<int> rfi_window_indices{3, 4, 6, 7, 8, 20, 30, 40};
    std::vector<std::complex<float>> samples;
    std::size_t sample_size = 40000;
    std::size_t window_size = 400;
    test_vector_generation(sample_size, window_size, 1, 10, 30, rfi_window_indices, samples);

    RFIStatistics stat;
    std::size_t nch = 1;
    sk_computation(nch, window_size, samples, stat);
    float expected_rfi_fraction = (rfi_window_indices.size()/float(sample_size/window_size)) + 0.01;
    EXPECT_EQ(expected_rfi_fraction, stat.rfi_fraction); //To check: fails inspite of actual and expected values being same.
}

TEST_F(SpectralKurtosisTester, sk_replacement)
{
   //Test vector
    bool with_rfi = 1;
    std::size_t sample_size = 40000;
    std::size_t window_size = 400;
    float rfi_freq = 10;
    float rfi_amp = 30;
    float mean =  5;
    float std = 2;
    SKTestVector tv(sample_size, window_size, with_rfi, rfi_freq, rfi_amp, mean, std);
    std::vector<int> rfi_window_indices{1, 2, 3, 4, 6, 7, 8, 9, 20, 30, 40};
    std::vector<std::complex<float>> samples;
    tv.generate_test_vector(rfi_window_indices, samples); //generating test vector
    
    //SK
    std::size_t nch = 1;
    float sk_min = 0.8;
    float sk_max = 1.2;
    RFIStatistics stat;
    SpectralKurtosis sk(nch, window_size, sk_min, sk_max);
    sk.compute_sk(samples, stat); //computing SK
    float expected_rfi_fraction = (rfi_window_indices.size()/float(sample_size/window_size)) + 0.01;
    EXPECT_EQ(expected_rfi_fraction, stat.rfi_fraction);

    //RFI replacement
    BOOST_LOG_TRIVIAL(info) <<"RFI replacement \n";
    SKRfiReplacement rr(stat.rfi_status);
    rr.replace_rfi_data(samples);

    //SK computation after RFI replacement
    BOOST_LOG_TRIVIAL(info) <<"computing SK after RFI replacement.. \n";
    sk.compute_sk(samples, stat); //computing SK
    float expected_val_after_rfi_replacement = 0;
    EXPECT_EQ(expected_val_after_rfi_replacement, stat.rfi_fraction);
}

} //test
} //edd
} //effelsberg
} //psrdada_cpp
