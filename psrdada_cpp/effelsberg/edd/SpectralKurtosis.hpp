#include "psrdada_cpp/common.hpp"
#include <complex>
#include <vector>
#include <numeric>

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {

#define DEFAULT_SK_MIN 0.9
#define DEFAULT_SK_MAX 1.1

struct RFIStatistics{
    std::vector<int>  rfi_status;
    float rfi_fraction;
};
   
class SpectralKurtosis{
public:
    /**
     * @param     nchannels     number of channels.
     *            window_size   number of samples per window.
     *            sk_min        minimum value of spectral kurtosis.
     *            sk_max        maximum value of spectral kurtosis.
     */
    SpectralKurtosis(int nchannels, int window_size, float sk_min = DEFAULT_SK_MIN,
		     float sk_max = DEFAULT_SK_MAX);
    ~SpectralKurtosis();

    /**
     * @brief     computes SK and returns rfi statistics.
     *
     */
    void compute_sk(std::vector<std::complex<float>> const& data, RFIStatistics& stats);

private:
    int _nchannels; //number of channels
    int _window_size; //window size
    int _nwindows; //number of windows
    int _sample_size; //size of input data
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

