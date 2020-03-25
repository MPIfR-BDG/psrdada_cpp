#include "psrdada_cpp/effelsberg/edd/SpectralKurtosis.hpp"

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {

SpectralKurtosis::SpectralKurtosis(int nchannels, int window_size)
    : _nchannels(nchannels),
      _window_size(window_size)
{
}

SpectralKurtosis::~SpectralKurtosis()
{
}

void SpectralKurtosis::init()
{
    if((_sample_size % _window_size) != 0){
        throw std::runtime_error("vector size is not a multiple of window_size. Give different window size\n");
    }	
    _nwindows = _sample_size / _window_size;
    _p1.resize(_sample_size);
    _p2.resize(_sample_size);
    _s1.resize(_sample_size);
    _s2.resize(_sample_size);
    _sk.resize(_sample_size);
}

void SpectralKurtosis::compute_sk(std::vector<std::complex<float>> data, RFIStatistics& stats)
{
    _sample_size = data.size();
    //initializing variables
    init();
    //computing sk
    float x;
    for(int i = 0; i < _sample_size; i++){
        x = abs(data[i]);
        _p1[i] = x * x; //power
        _p2[i] = _p1[i] * _p1[i];
    }
    stats.rfi_status.resize(_nwindows * _nchannels);
    std::size_t r1, r2;
    for(int i = 0; i < _nwindows; i++){
        r1 = i * _window_size;
     	r2 = r1 + _window_size;
	_s1[i] = std::accumulate((_p1.begin() + r1), (_p1.begin() + r2), 0);
	_s2[i] = std::accumulate((_p2.begin() + r1), (_p2.begin() + r2), 0);
	_sk[i] = ((_window_size + 1) / (_window_size - 1)) *
		(((_window_size * _s2[i]) / (_s1[i] * _s1[i])) - 1);
	if((_sk[i] > SK_MAX) || (_sk[i] < SK_MIN)){
            stats.rfi_status[i] = 1;
        }
	else{
            stats.rfi_status[i] = 0;
        }
    }
    stats.rfi_fraction = std::accumulate(stats.rfi_status.begin(), stats.rfi_status.end(), 0.0) / _nwindows;
    BOOST_LOG_TRIVIAL(info) << "RFI fraction: " << stats.rfi_fraction;
}
} //edd
} //effelsberg
} //psrdada_cpp
