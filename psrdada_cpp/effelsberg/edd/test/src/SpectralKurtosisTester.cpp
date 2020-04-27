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
    SKTestVector tv(sample_size, window_size, with_rfi, rfi_freq, rfi_amp);
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
    test_vector_generation(4000, 150, 0, 0, 0, rfi_window_indices, samples);

    RFIStatistics stat;
    try{
        sk_computation(1, 150, samples, stat);
	FAIL() << "Expected std::runtime_error\n";
    }
    catch(std::runtime_error const & err){
        EXPECT_EQ(err.what(), std::string("sample size is not a multiple of window_size. Give different window size\n"));
    }
}

TEST_F(SpectralKurtosisTester, sk_withoutRFI)
{
    std::vector<int> rfi_window_indices{};
    std::vector<std::complex<float>> samples;
    test_vector_generation(40000, 400, 0, 0, 0, rfi_window_indices, samples);

    RFIStatistics stat;
    sk_computation(1, 400, samples, stat);
    float expected_rfi_fraction = 0.01;
    EXPECT_EQ(expected_rfi_fraction, stat.rfi_fraction);
}

TEST_F(SpectralKurtosisTester, sk_withRFI)
{
    std::vector<int> rfi_window_indices{3, 4, 6, 7, 8, 20, 30, 40};
    std::vector<std::complex<float>> samples;
    test_vector_generation(40000, 400, 1, 10, 1, rfi_window_indices, samples);

    RFIStatistics stat;
    sk_computation(1, 400, samples, stat);
    float expected_rfi_fraction = (rfi_window_indices.size()/float(40000/400)) + 0.01;
    EXPECT_EQ(expected_rfi_fraction, stat.rfi_fraction);
    //float expected_rfi_fraction = (rfi_window_indices.size()/float(40000/400)) + 0.01;
    //EXPECT_EQ(expected_rfi_fraction, stat.rfi_fraction);
}

TEST_F(SpectralKurtosisTester, sk_replacement)
{
   //Test vector
    bool with_rfi = 1;
    std::size_t sample_size = 40000;
    std::size_t window_size = 400;
    float rfi_freq = 10;
    float rfi_amp = 1;
    SKTestVector tv(sample_size, window_size, with_rfi, rfi_freq, rfi_amp);
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
    float expected_rfi_fraction = (rfi_window_indices.size()/float(40000/400)) + 0.01;
    EXPECT_EQ(expected_rfi_fraction, stat.rfi_fraction);

    //RFI replacement
    BOOST_LOG_TRIVIAL(info) <<"RFI replacement \n";
    SKRfiReplacement rr(samples, stat.rfi_status);
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
