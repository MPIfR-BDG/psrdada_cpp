#include "psrdada_cpp/effelsberg/edd/SpectralKurtosis.hpp"

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {

SpectralKurtosis::SpectralKurtosis(std::size_t nchannels, std::size_t window_size, float sk_min, float sk_max)
    : _nchannels(nchannels),
      _window_size(window_size),
      _sk_min(sk_min),
      _sk_max(sk_max)
{
    BOOST_LOG_TRIVIAL(debug) << "Creating new SpectralKurtosis instance..\n";    
}

SpectralKurtosis::~SpectralKurtosis()
{
    BOOST_LOG_TRIVIAL(debug) << "Destroying SpectralKurtosis instance..\n";    
}

void SpectralKurtosis::init()
{
    if((_sample_size % _window_size) != 0){
        BOOST_LOG_TRIVIAL(error) << "Sample size " << _sample_size <<" is not a multiple of window_size "
        << _window_size <<". Give different window size.\n";
        throw std::runtime_error("sample size is not a multiple of window_size. Give different window size\n");
    }
    _nwindows = _sample_size / _window_size;
    _p1.resize(_sample_size);
    _p2.resize(_sample_size);
    _s1.resize(_sample_size);
    _s2.resize(_sample_size);
    _sk.resize(_sample_size);
}

void SpectralKurtosis::compute_sk(std::vector<std::complex<float>> const& data, RFIStatistics& stats)
{
    _sample_size = data.size();
    BOOST_LOG_TRIVIAL(debug) << "Computing SK for sample_size " << _sample_size
    << " and window_size " << _window_size <<"\n";
    //initializing variables
    init();
    //computing sk
    for(std::size_t samp_idx = 0; samp_idx < _sample_size; samp_idx++){
        float x = abs(data[samp_idx]);
        _p1[samp_idx] = x * x; //power
        _p2[samp_idx] = _p1[samp_idx] * _p1[samp_idx];
    }
    stats.rfi_status.resize(_nwindows * _nchannels);
    float sk_factor = (_window_size + 1) / (_window_size - 1);
    for(std::size_t window_idx = 0; window_idx < _nwindows; window_idx++){
        std::size_t r1 = window_idx * _window_size;
        std::size_t r2 = r1 + _window_size;
        _s1[window_idx] = std::accumulate((_p1.begin() + r1), (_p1.begin() + r2), 0);
        _s2[window_idx] = std::accumulate((_p2.begin() + r1), (_p2.begin() + r2), 0);
	//computing SK for the window
        _sk[window_idx] = sk_factor * (((_window_size * _s2[window_idx]) / (_s1[window_idx] * _s1[window_idx])) - 1);
	//Updating RFI status for that window
        stats.rfi_status[window_idx] = ((_sk[window_idx] > _sk_max) || (_sk[window_idx] < _sk_min));
    }
    stats.rfi_fraction = std::accumulate(stats.rfi_status.begin(), stats.rfi_status.end(), 0.0) / _nwindows;
    BOOST_LOG_TRIVIAL(info) << "RFI fraction: " << stats.rfi_fraction;
}
} //edd
} //effelsberg
} //psrdada_cpp
