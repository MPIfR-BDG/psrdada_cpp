#include "psrdada_cpp/effelsberg/edd/SpectralKurtosis.cuh"

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {

struct compute_power{
    __host__ __device__
    float operator()(const thrust::complex<float> &z)
    {
        return (thrust::abs(z) * thrust::abs(z));
    }
};

struct square{
    __host__ __device__
    float operator()(const float &z)
    {
        return (z * z);
    }
};

struct computing_sk{
    const std::size_t M; //_window_size
    computing_sk(std::size_t _M) : M(_M) {}

    __host__ __device__
    float operator() (const float &s1, const float &s2) const {
        return  ((M + 1) / (M - 1)) * (((M * s2) / (s1 * s1)) - 1);
   }
};

struct check_sk_thresholds{
    const float sk_min;
    const float sk_max;
    check_sk_thresholds(float min, float max)
        : sk_min(min),
          sk_max(max)
    {}

    __host__ __device__
    int operator() (const float &sk) const {
        return ((sk < sk_min) || (sk > sk_max)) ;
   }
};

struct set_indices{
    const std::size_t M; //window_size
    set_indices(std::size_t _M) : M(_M) {}
    
    __host__ __device__
    int operator()(const int &z)
    {
        return (z / M);
    }
};

SpectralKurtosis::SpectralKurtosis(std::size_t nchannels, std::size_t window_size, float sk_min, float sk_max)
    : _nchannels(nchannels),
      _window_size(window_size),
      _sk_min(sk_min),
      _sk_max(sk_max)
{
    BOOST_LOG_TRIVIAL(debug) << "Creating new SpectralKurtosis instance... \n";
}

SpectralKurtosis::~SpectralKurtosis()
{
    BOOST_LOG_TRIVIAL(debug) << "Destroying SpectralKurtosis instance... \n";
}

void SpectralKurtosis::init()
{
    if((_sample_size % _window_size) != 0){
        BOOST_LOG_TRIVIAL(error) << "Sample(data) size " << _sample_size <<" is not a multiple of _window_size "
                                 << _window_size <<". Give different window size.\n";
        throw std::runtime_error("Data(sample) size is not a multiple of window_size. Give different window size\n");
    }
    _nwindows = _sample_size /_window_size;
    _d_p1.resize(_sample_size);
    _d_p2.resize(_sample_size);
    _d_s1.resize(_nwindows);
    _d_s2.resize(_nwindows);
    _d_sk.resize(_nwindows);
}

void SpectralKurtosis::compute_sk(const thrust::device_vector<thrust::complex<float>> &data, RFIStatistics &stats){
    _sample_size = data.size();
    BOOST_LOG_TRIVIAL(debug) << "Computing SK for sample_size " << _sample_size
                             << " and window_size " << _window_size <<".\n";
    //initializing class variables
    init();
    //computing SK
    thrust::transform(data.begin(), data.end(), _d_p1.begin(), compute_power());
    thrust::transform(_d_p1.begin(), _d_p1.end(), _d_p2.begin(), square());
    //computing sum of _d_p1 for all windows
    thrust::reduce_by_key(thrust::device, 
                          thrust::make_transform_iterator(thrust::counting_iterator<int> (0), set_indices(_window_size)),
                          thrust::make_transform_iterator(thrust::counting_iterator<int> ((_sample_size - 1)), set_indices(_window_size)), 
                          _d_p1.begin(), 
                          thrust::discard_iterator<int>(), 
                          _d_s1.begin());
    //computing sum of _d_p2 for all windows
    thrust::reduce_by_key(thrust::device, 
                          thrust::make_transform_iterator(thrust::counting_iterator<int> (0), set_indices(_window_size)),
                          thrust::make_transform_iterator(thrust::counting_iterator<int> ((_sample_size - 1)), set_indices(_window_size)), 
                          _d_p2.begin(), 
                          thrust::discard_iterator<int>(), 
                          _d_s2.begin());
    //computing sk for all windows
    thrust::transform(_d_s1.begin(), _d_s1.end(), _d_s2.begin(), _d_sk.begin(), computing_sk(_window_size));
    //updating RFI status
    stats.rfi_status.resize(_nwindows);
    thrust::transform(_d_sk.begin(), _d_sk.end(), stats.rfi_status.begin(), check_sk_thresholds(_sk_min, _sk_max));
    stats.rfi_fraction = thrust::reduce(stats.rfi_status.begin(), stats.rfi_status.end(), 0.0f) / _nwindows;
    BOOST_LOG_TRIVIAL(info) << "RFI fraction: " << stats.rfi_fraction;
}

} //edd
} //effelsberg
} //psrdada_cpp


