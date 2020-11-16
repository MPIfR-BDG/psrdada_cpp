#ifndef PSRDADA_CPP_EFFELSBERG_EDD_SPECTRALKURTOSIS_HPP
#define PSRDADA_CPP_EFFELSBERG_EDD_SPECTRALKURTOSIS_HPP

#include "psrdada_cpp/common.hpp"
#include <complex>
#include <vector>
#include <numeric>

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {

struct RFIStatistics{
    std::vector<int>  rfi_status;
    float rfi_fraction;
};
   
class SpectralKurtosis{
public:
    /**
     * @brief     constructor
     *
     * @param     nchannels     number of channels.
     *            window_size   number of samples per window.
     *            sk_min        minimum value of spectral kurtosis.
     *            sk_max        maximum value of spectral kurtosis.
     */
    SpectralKurtosis(std::size_t nchannels, std::size_t window_size, float sk_min = 0.9,
		     float sk_max = 1.1);
    ~SpectralKurtosis();

    /**
     * @brief     computes spectral kurtosis for the given data and returns its rfi statistics.
     *
     * @param     data          input data
     *            stats         RFI statistics
     *
     */
    void compute_sk(std::vector<std::complex<float>> const& data, RFIStatistics& stats);

private:
    std::size_t _nchannels; //number of channels
    std::size_t _window_size; //window size
    std::size_t _nwindows; //number of windows
    std::size_t _sample_size; //size of input data
    float _sk_min, _sk_max;
    std::vector<float> _p1, _p2, _s1, _s2, _sk;

    /**
     * @brief     initializes the data members of the class.
     *
     * */
    void init();
};
} //edd
} //effelsberg
} //psrdada_cpp

#endif //PSRDADA_CPP_EFFELSBERG_EDD_SPECTRALKURTOSIS_HPP
