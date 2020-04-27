#include "psrdada_cpp/common.hpp"

#include <complex>
#include <vector>
#include <functional>
#include <numeric>
#include <random>
#include <algorithm>

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {
namespace test {

#define DEFAULT_MEAN 0 //default mean for normal ditribution test vector
#define DEFAULT_STD 0.5 //default standard deviation for normal ditribution test vector

class SKTestVector{
public:
    /**
     * @param    sample_size        size of test vector
     *           window_size        number of samples in a window
     *           with_rfi           Flag to include RFI in test vector
     *           rfi_frequency      frequency of RFI
     *           rfi_amplitude      amplitude of RFI
     *           mean               mean for normal distribution test vector
     *           std                standard deviation for normal distribution test vector
     */           
    SKTestVector(std::size_t sample_size, std::size_t window_size, bool with_rfi, float rfi_frequency, 
		 float rfi_amplitude, float mean = DEFAULT_MEAN, float std = DEFAULT_STD);
    ~SKTestVector();
    /**
     * @brief    generates test vector
     *
     * @detail   The test vector is a normal distribution vector and contains RFI if the flag with_rfi is set to true.
     *
     * @param    rfi_windows        vector of window indices on which the RFI has to be added.
     *           test_samples       output test vector
     */           
    void generate_test_vector(std::vector<int> const& rfi_windows, std::vector<std::complex<float>> &test_samples);
private:
    /**
     * @brief    generates a normal distribution vector for the default or given mean and standard deviation.
     *
     * @param    samples            output normal distribution test vector
     *           
     */           
    void generate_normal_distribution_vector(std::vector<std::complex<float>> &samples);
    /**
     * @brief    generates rfi signal of frequency = _rfi_frequency and size = _window_size
     *
     * @param    rfi_samples       output RFI vector
     *
     */
    void generate_rfi_vector(std::vector<std::complex<float>> &rfi_samples);
    std::size_t _sample_size;
    std::size_t _window_size;
    bool _with_rfi;
    float _rfi_frequency;
    float _rfi_amplitude;
    float _mean;
    float _std;
};

} //test
} //edd
} //effelsberg
} //psrdada_cpp

