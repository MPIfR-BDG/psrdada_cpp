#include "psrdada_cpp/common.hpp"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/complex.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/execution_policy.h>

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {

#define DEFAULT_SK_MIN 0.9
#define DEFAULT_SK_MAX 1.1

struct RFIStatistics{
    thrust::device_vector<int> rfi_status;
    float rfi_fraction;
};

class SpectralKurtosisCuda{
public:
    /**
     * @brief      constructor
     *
     * @param(in)  nchannels     number of channels.
     * @param(in)  window_size   number of samples per window.
     * @param(in)  sk_min        minimum value of spectral kurtosis.
     * @param(in)  sk_max        maximum value of spectral kurtosis.
     */
    SpectralKurtosisCuda(std::size_t nchannels, std::size_t window_size, float sk_min = DEFAULT_SK_MIN,
                         float sk_max = DEFAULT_SK_MAX);
    ~SpectralKurtosisCuda();

    /**
     * @brief      computes spectral kurtosis for the given data and returns its rfi statistics.
     *
     * @param(in)  data          input data
     * @param(out) stats         RFI statistics
     *
     */
    void compute_sk(thrust::device_vector<thrust::complex<float>> const& data, RFIStatistics &stats);

private:
    /**
     * @brief     initializes the data members of the class.
     *
     * */
    void init();
    std::size_t _nchannels; //number of channels
    std::size_t _window_size; //window size
    std::size_t _nwindows; //number of windows
    std::size_t _sample_size; //size of input data
    float _sk_min, _sk_max;
    thrust::device_vector<float> _d_p1, _d_s1, _d_s2;
};
} //edd
} //effelsberg
} //psrdada_cpp
