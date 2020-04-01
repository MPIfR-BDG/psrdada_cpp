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

#define MEAN 1
#define STD 0.5

class SKTestVector{
public:
    /**
     * @param    sample_size        size of test vector
     *           window_size        number os samples in a window
     *           with_rfi           Flag to compute SK with or without RFI
     */           
    SKTestVector(int sample_size, int window_size, bool with_rfi);
    ~SKTestVector();
    /**
     * @brief    generates a test vector which is a normal distribution vector containing RFI if with_rfi is rfi. 
     *
     * @param    rfi_windows        window indices that has RFI
     *           test_samples       test vector of size sample_size
     */           
    void generate_test_vector(std::vector<int> rfi_windows, std::vector<std::complex<float>> &test_samples);
private:
    /**
     * @brief    generates a normal distribution vector of size sample_size for mean = 1 and standard deviation = 0.5
     *
     * @param    samples            normal distribution vector
     *           
     */           
    void generate_normal_distribution_vector(std::vector<std::complex<float>> &samples);
    /**
     * @brief    generates sine wave vector of length = _window_size
     *
     * @param    sine_samples       sine vector
     *
     */
    void generate_sine_vector(std::vector<std::complex<float>> &sine_samples);
    int _sample_size;
    int _window_size;
    bool _with_rfi;
};

} //test
} //edd
} //effelsberg
} //psrdada_cpp

