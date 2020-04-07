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

TEST_F(SpectralKurtosisTester, sk_window_size_chk)
{
    bool with_rfi = 0;
    std::size_t sample_size = 2000;
    std::size_t window_size = 150;
    SKTestVector tv(sample_size, window_size, with_rfi);
    std::vector<int> rfi_ind{};
    std::vector<std::complex<float>> samples;
    tv.generate_test_vector(rfi_ind, samples);
    
    std::size_t nch = 1;
    SpectralKurtosis sk(nch, window_size);
    RFIStatistics stat;
    sk.compute_sk(samples, stat);
}
TEST_F(SpectralKurtosisTester, sk_withoutRFI)
{
    bool with_rfi = 0;
    std::size_t sample_size = 20000;
    std::size_t window_size = 200;
    SKTestVector tv(sample_size, window_size, with_rfi);
    std::vector<int> rfi_ind{};
    std::vector<std::complex<float>> samples;
    tv.generate_test_vector(rfi_ind, samples); //generate test vector
    
    std::size_t nch = 1;
    float sk_min = 0.8;
    float sk_max = 1.2;
    SpectralKurtosis sk(nch, window_size, sk_min, sk_max);
    RFIStatistics stat;
    sk.compute_sk(samples, stat); //computing SK
}

TEST_F(SpectralKurtosisTester, sk_withRFI)
{
    bool with_rfi = 1;
    std::size_t sample_size = 40000;
    std::size_t window_size = 400;
    SKTestVector tv(sample_size, window_size, with_rfi);
    std::vector<int> rfi_ind{1, 2, 5, 7, 9, 20, 30, 40};
    std::vector<std::complex<float>> samples;
    tv.generate_test_vector(rfi_ind, samples); //generating test vector

    std::size_t nch = 1;
    float sk_min = 0.8;
    float sk_max = 1.2;
    SpectralKurtosis sk(nch, window_size, sk_min, sk_max);
    RFIStatistics stat;
    sk.compute_sk(samples, stat); //computing SK
    BOOST_LOG_TRIVIAL(info) <<"RFI status: ";
    for(int ii = 0; ii< 10; ii++){
        printf("RFI[%d] = %d\n", ii, stat.rfi_status[ii]);
    }
}

} //test
} //edd
} //effelsberg
} //psrdada_cpp




