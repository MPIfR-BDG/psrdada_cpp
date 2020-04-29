#include "psrdada_cpp/effelsberg/edd/SKRfiReplacement.hpp"

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {

SKRfiReplacement::SKRfiReplacement(const std::vector<int> &rfi_status)
    :  _rfi_status(rfi_status)
{
    BOOST_LOG_TRIVIAL(info) << "Creating new SKRfiReplacement instance..\n";
}

SKRfiReplacement::~SKRfiReplacement()
{
    BOOST_LOG_TRIVIAL(info) << "Destroying SKRfiReplacement instance..\n";
}

void SKRfiReplacement::init()
{
    BOOST_LOG_TRIVIAL(info) << "initializing data_members of SKRfiReplacement class..\n";
    _nwindows = _rfi_status.size();
    _rfi_window_indices.reserve(_nwindows);
    get_rfi_window_indices();
    _clean_window_indices.reserve(_nwindows);
    get_clean_window_indices();
}

void SKRfiReplacement::get_rfi_window_indices()
{
    _nrfi_windows = std::count(_rfi_status.begin(), _rfi_status.end(), 1);
    _rfi_window_indices.resize(_nrfi_windows);
    std::size_t iter = 0;
    for(std::size_t index = 0; index < _nrfi_windows; index++){
        _rfi_window_indices[index] = std::distance(_rfi_status.begin(), 
                                     max_element((_rfi_status.begin() + iter), _rfi_status.end()));
	iter = _rfi_window_indices[index] + 1;
    }
}

void SKRfiReplacement::get_clean_window_indices()
{
    _nclean_windows = std::count(_rfi_status.begin(), _rfi_status.end(), 0);
    _clean_window_indices.resize(DEFAULT_NUM_CLEAN_WINDOWS);
    std::size_t iter = 0;
    for(std::size_t index = 0; index < DEFAULT_NUM_CLEAN_WINDOWS; index++){
        _clean_window_indices[index] = std::distance(_rfi_status.begin(), 
                                       min_element((_rfi_status.begin() + iter), _rfi_status.end()));
	iter = _clean_window_indices[index] + 1;
    }
}

void SKRfiReplacement::get_clean_data_statistics(const std::vector<std::complex<float>> &data,
                                                 DataStatistics &ref_data_statistics)
{
    _window_size = data.size() / _nwindows;
    std::vector<std::complex<float>> clean_data(DEFAULT_NUM_CLEAN_WINDOWS * _window_size);
    for(std::size_t ii = 0; ii < DEFAULT_NUM_CLEAN_WINDOWS; ii++){
        std::size_t window_index = _clean_window_indices[ii];
	std::size_t ibegin = window_index * _window_size;
	std::size_t iend = ibegin + _window_size - 1;
	std::size_t jj = ii * _window_size;
	std::copy((data.begin() + ibegin), (data.begin() + iend), (clean_data.begin() + jj));
        BOOST_LOG_TRIVIAL(debug) <<"clean_win_index = " << window_index
                                 << " ibegin = " << ibegin << " iend = " << iend;
    }
    compute_data_statistics(clean_data, ref_data_statistics);
}

void SKRfiReplacement::compute_data_statistics(const std::vector<std::complex<float>> &data, DataStatistics &stats)
{
    std::size_t length = data.size();
    std::complex<float> sum = std::accumulate(data.begin(), data.end(), std::complex<float> (0, 0));
    stats.r_mean = sum.real() / length;
    stats.i_mean = sum.imag() / length;
    std::vector<float> vreal(length), vimag(length), rdiff(length), idiff(length);
    for(std::size_t ii = 0; ii < length; ii++){
        vreal[ii] = data[ii].real();
	vimag[ii] = data[ii].imag();
    }
    std::transform(vreal.begin(), vreal.end(), rdiff.begin(), std::bind2nd(std::minus<float>(), stats.r_mean));
    std::transform(vimag.begin(), vimag.end(), idiff.begin(), std::bind2nd(std::minus<float>(), stats.i_mean));
    stats.r_sd = std::sqrt((float)std::inner_product(rdiff.begin(), rdiff.end(), rdiff.begin(), 0.0f) / length);
    stats.i_sd = std::sqrt((float)std::inner_product(idiff.begin(), idiff.end(), idiff.begin(), 0.0f) / length);
    BOOST_LOG_TRIVIAL(debug) << "DataStatistics r_mean = " << stats.r_mean
                             << " r_sd =  " << stats.r_sd
                             << " i_mean = " << stats.i_mean
                             << " i_sd = " << stats.r_sd;
}

void SKRfiReplacement::generate_replacement_data(const DataStatistics &stats, std::vector<std::complex<float>> &replacement_data)
{
    BOOST_LOG_TRIVIAL(info) << "generating replacement data..\n";
    replacement_data.resize(_window_size);
    std::default_random_engine gen(1);
    std::normal_distribution<float> rdist(stats.r_mean, stats.r_sd);
    std::normal_distribution<float> idist(stats.i_mean, stats.i_sd);
    for(std::size_t index = 0; index < _window_size; index++){
        replacement_data[index] = std::complex<float>(rdist(gen), idist(gen));
    }
}

void SKRfiReplacement::replace_rfi_data(std::vector<std::complex<float>> &data)
{
    DataStatistics stats;
    std::vector<std::complex<float>> replacement_data;
    //initialize data members of the class
    init();
    //RFI present and not in all windows
    if((_nrfi_windows > 0) && (_nrfi_windows < _nwindows)){
        get_clean_data_statistics(data, stats);
        generate_replacement_data(stats, replacement_data);
	//Replacing RFI
	for(std::size_t ii = 0; ii < _nrfi_windows; ii++){
            std::size_t index = _rfi_window_indices[ii] * _window_size;
	    std::copy(replacement_data.begin(), replacement_data.end(), (data.begin() +index));
	}
    }
}

} //edd
} //effelsberg
} //psrdada_cpp
