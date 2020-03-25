#include "psrdada_cpp/common.hpp"
#include <complex>
#include <vector>
#include <numeric>

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {

#define SK_MIN 0.9
#define SK_MAX 1.1

struct RFIStatistics{
    std::vector<int>  rfi_status;
    float rfi_fraction;
};
   
class SpectralKurtosis{
private:
    int _nchannels; //number of channels
    int _window_size; //window size
    int _nwindows; //number of windows
    int _sample_size; //size of input data
    std::vector<float> _p1, _p2, _s1, _s2, _sk;

public:
    /**
     * @param     nch           number of channels
     *            window_size   number of samples per window.
     */
    //SpectralKurtosis(std::vector<std::complex<float>> data, int nchannels, int window_size);
    SpectralKurtosis(int nchannels, int window_size);
    ~SpectralKurtosis();
    /**
     * @brief     initializes data members of the class.
     *
     * @param     sample_size    size of the input data
     */
    void init();
    /**
     * @brief     computes SK and returns rfi statistics.
     *
     */
    void compute_sk(std::vector<std::complex<float>> data, RFIStatistics& stats);
};
} //edd
} //effelsberg
} //psrdada_cpp

