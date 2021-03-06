#include "psrdada_cpp/effelsberg/edd/test/SpectralKurtosisCudaTester.cuh"

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {
namespace test {

SpectralKurtosisCudaTester:: SpectralKurtosisCudaTester():
       	::testing::Test()
{
}

SpectralKurtosisCudaTester::~SpectralKurtosisCudaTester()
{
}

void SpectralKurtosisCudaTester::SetUp()
{
}

void SpectralKurtosisCudaTester::TearDown()
{
}

void SpectralKurtosisCudaTester::test_vector_generation(std::size_t sample_size, std::size_t window_size, 
		                                    bool with_rfi, float rfi_freq, float rfi_amp,
						    const std::vector<int> &rfi_window_indices,
						    std::vector<std::complex<float>> &samples)
{
    float m = 0;
    float std = 1;
    SKTestVector tv(sample_size, window_size, with_rfi, rfi_freq, rfi_amp, m, std);
    tv.generate_test_vector(rfi_window_indices, samples);
}

void SpectralKurtosisCudaTester::sk_computation(std::size_t nch, std::size_t window_size, 
                                           const std::vector<std::complex<float>> &samples,
					   RFIStatistics &stat)
{
    thrust::host_vector<thrust::complex<float>> h_samples(samples);
    thrust::device_vector<thrust::complex<float>> d_samples(h_samples);
    float sk_min = 0.8;
    float sk_max = 1.2;    
    SpectralKurtosisCuda sk(nch, window_size, sk_min, sk_max);
    sk.compute_sk(d_samples, stat);
}

TEST_F(SpectralKurtosisCudaTester, sk_window_size_check)
{
    std::vector<int> rfi_window_indices{};
    std::vector<std::complex<float>> samples;
    std::size_t sample_size = 4000;
    std::size_t window_size = 150;
    bool with_rfi = 0;
    float rfi_freq = 0;
    float rfi_amp = 0;
    test_vector_generation(sample_size, window_size, with_rfi, rfi_freq, rfi_amp, rfi_window_indices, samples);

    RFIStatistics stat;
    std::size_t nch = 1;
    EXPECT_THROW(sk_computation(nch, window_size, samples, stat), std::runtime_error);
}

TEST_F(SpectralKurtosisCudaTester, sk_withoutRFI)
{
    std::vector<int> rfi_window_indices{};
    std::vector<std::complex<float>> samples;
    std::size_t sample_size = 40000;
    std::size_t window_size = 400;
    bool with_rfi = 0;
    float rfi_freq = 0;
    float rfi_amp = 0;
    test_vector_generation(sample_size, window_size, with_rfi, rfi_freq, rfi_amp, rfi_window_indices, samples);

    RFIStatistics stat;
    std::size_t nch = 1;
    sk_computation(nch, window_size, samples, stat);
    float expected_rfi_fraction = 0;
    EXPECT_EQ(expected_rfi_fraction, stat.rfi_fraction);
}

TEST_F(SpectralKurtosisCudaTester, sk_withRFI)
{
    std::vector<int> rfi_window_indices{3, 4, 6, 7, 8, 20, 30, 40, 45, 75};
    std::vector<std::complex<float>> samples;
    std::size_t sample_size = 40000;
    std::size_t window_size = 400;
    bool with_rfi = 1;
    float rfi_freq = 30;
    float rfi_amp = 10;
    test_vector_generation(sample_size, window_size, with_rfi, rfi_freq, rfi_amp, rfi_window_indices, samples);

    RFIStatistics stat;
    std::size_t nch = 1;
    sk_computation(nch, window_size, samples, stat);
    float expected_rfi_fraction = (rfi_window_indices.size()/float(sample_size/window_size));
    EXPECT_EQ(expected_rfi_fraction, stat.rfi_fraction);
}

TEST_F(SpectralKurtosisCudaTester, sk_RFIreplacement)
{
    std::size_t sample_size = 128* 1024 * 1024;
    std::size_t window_size = 1024 * 2;
    //Test vector generation
    std::vector<int> rfi_window_indices{3, 4, 6, 7, 8, 20, 30, 40, 45, 75};
    std::vector<std::complex<float>> samples;
    bool with_rfi = 1;
    float rfi_freq = 30;
    float rfi_amp = 10;
    test_vector_generation(sample_size, window_size, with_rfi, rfi_freq, rfi_amp, rfi_window_indices, samples);

    //SK computation
    thrust::host_vector<thrust::complex<float>> h_samples(samples);
    thrust::device_vector<thrust::complex<float>> d_samples(h_samples);
    float sk_min = 0.8;
    float sk_max = 1.2;    
    std::size_t nch = 1;
    SpectralKurtosisCuda sk(nch, window_size, sk_min, sk_max);
    RFIStatistics stat;
    sk.compute_sk(d_samples, stat);

    //RFI replacement
    BOOST_LOG_TRIVIAL(info) <<"RFI replacement..\n";
    SKRfiReplacementCuda rr;
    rr.replace_rfi_data(stat.rfi_status, d_samples);

    //SK computation after RFI replacement
    BOOST_LOG_TRIVIAL(info) <<"computing SK after replacing the RFI data..\n";
    sk.compute_sk(d_samples, stat);
    float expected_val_after_rfi_replacement = 0;
    EXPECT_EQ(expected_val_after_rfi_replacement, stat.rfi_fraction);
}

TEST_F(SpectralKurtosisCudaTester, sk_kernel)
{
    std::size_t sample_size = 160000000;
    std::size_t window_size = 2000;
    std::size_t nwindows = sample_size / window_size;
    //Test vector generation
    std::vector<int> rfi_window_indices{1, 4, 6, 7, 8, 20, 30, 40, 45, 75};
    std::vector<std::complex<float>> samples;
    bool with_rfi = 1;
    float rfi_freq = 30;
    float rfi_amp = 10;
    test_vector_generation(sample_size, window_size, with_rfi, rfi_freq, rfi_amp, rfi_window_indices, samples);

    //SK computation
    thrust::host_vector<thrust::complex<float>> h_samples(samples);
    thrust::device_vector<thrust::complex<float>> d_samples(h_samples);

    std::size_t nch = 1;    
    SpectralKurtosisCuda sk(nch, window_size);
    RFIStatistics stat, stat_k;
    sk.compute_sk_thrust(d_samples, stat);
    sk.compute_sk(d_samples, stat_k);
    for (int ii = 0; ii < nwindows; ii++){
        EXPECT_EQ(stat.rfi_status[ii], stat_k.rfi_status[ii]);
    }
    EXPECT_EQ(stat.rfi_fraction, stat_k.rfi_fraction);
    
    //RFI replacement
    BOOST_LOG_TRIVIAL(info) <<"RFI replacement..\n";
    SKRfiReplacementCuda rr;
    std::size_t clean_windows = 100; //no. of clean windows used for computing data statistics
    rr.replace_rfi_data(stat_k.rfi_status, d_samples, clean_windows);

    //SK computation after RFI replacement
    BOOST_LOG_TRIVIAL(info) <<"computing SK after replacing the RFI data..\n";
    sk.compute_sk(d_samples, stat_k);
    float expected_val_after_rfi_replacement = 0;
    EXPECT_EQ(expected_val_after_rfi_replacement, stat_k.rfi_fraction);
}

} //test
} //edd
} //effelsberg
} //psrdada_cpp
