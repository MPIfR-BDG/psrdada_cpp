#include "psrdada_cpp/effelsberg/edd/SKRfiReplacementCuda.cuh"

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {

struct get_real{
    __host__ __device__
    float operator() (thrust::complex<float> val) const{
        return val.real();
    }
};

struct get_imag{
    __host__ __device__
    float operator() (thrust::complex<float> val) const{
        return val.imag();
    }
};

struct mean_subtraction_square{
    const float mean;
    mean_subtraction_square(float _mean) :mean(_mean) {}
    __host__ __device__
    float operator() (float val) const{
        return ((val - mean) * (val - mean));
    }
};

struct equals_one{
    __host__ __device__
    float operator() (int val) const{
        return (val == 1);
    }
};

struct equals_zero{
    __host__ __device__
    float operator() (int val) const{
        return (val == 0);
    }
};

SKRfiReplacementCuda::SKRfiReplacementCuda()
{
    BOOST_LOG_TRIVIAL(info) << "Creating new SKRfiReplacementCuda instance..\n";
}

SKRfiReplacementCuda::~SKRfiReplacementCuda()
{
    BOOST_LOG_TRIVIAL(info) << "Destroying SKRfiReplacementCuda instance..\n";
}

void SKRfiReplacementCuda::init()
{
    BOOST_LOG_TRIVIAL(info) << "initializing data_members of SKRfiReplacementCuda class..\n";
    _nwindows = _rfi_status.size();
    _rfi_window_indices.reserve(_nwindows);
    get_rfi_window_indices();
    _clean_window_indices.reserve(_nwindows);
    get_clean_window_indices();
}

void SKRfiReplacementCuda::get_rfi_window_indices()
{
    _nrfi_windows = thrust::count(_rfi_status.begin(), _rfi_status.end(), 1);
    _rfi_window_indices.resize(_nrfi_windows);
    thrust::copy_if(thrust::make_counting_iterator<int>(0),
		    thrust::make_counting_iterator<int>(_nwindows),
		    _rfi_status.begin(),
		    _rfi_window_indices.begin(),
		    equals_one());
}

void SKRfiReplacementCuda::get_clean_window_indices()
{
    _nclean_windows = thrust::count(_rfi_status.begin(), _rfi_status.end(), 0);
    _clean_window_indices.resize(DEFAULT_NUM_CLEAN_WINDOWS);
    thrust::copy_if(thrust::make_counting_iterator<int>(0),
		    thrust::make_counting_iterator<int>(_nwindows),
		    _rfi_status.begin(),
		    _clean_window_indices.begin(),
		    equals_zero());
}

void SKRfiReplacementCuda::get_clean_data_statistics(const thrust::device_vector<thrust::complex<float>> &data,
                                                     DataStatistics &ref_data_statistics)
{
    _window_size = data.size() / _nwindows;
    thrust::device_vector<thrust::complex<float>> clean_data(DEFAULT_NUM_CLEAN_WINDOWS * _window_size);
    for(std::size_t ii = 0; ii < DEFAULT_NUM_CLEAN_WINDOWS; ii++){
        std::size_t window_index = _clean_window_indices[ii];
        std::size_t ibegin = window_index * _window_size;
        std::size_t iend = ibegin + _window_size - 1;
        std::size_t jj = ii * _window_size;
        thrust::copy((data.begin() + ibegin), (data.begin() + iend), (clean_data.begin() + jj));
        BOOST_LOG_TRIVIAL(debug) <<"clean_win_index = " << window_index
                                 << " ibegin = " << ibegin << " iend = " << iend;
    }
    compute_data_statistics(clean_data, ref_data_statistics);
}

void SKRfiReplacementCuda::compute_data_statistics(const thrust::device_vector<thrust::complex<float>> &data, 
                                                   DataStatistics &stats)
{
    std::size_t length = data.size();
    _d_vreal.resize(length);
    _d_vimag.resize(length);
    thrust::transform(data.begin(), data.end(), _d_vreal.begin(), get_real());
    thrust::transform(data.begin(), data.end(), _d_vimag.begin(), get_imag());
    stats.r_mean = thrust::reduce(_d_vreal.begin(), _d_vreal.end(), 0.0f) / length;
    stats.i_mean = thrust::reduce(_d_vimag.begin(), _d_vimag.end(), 0.0f) / length;
    stats.r_sd = std::sqrt(thrust::transform_reduce(_d_vreal.begin(), _d_vreal.end(), mean_subtraction_square(stats.r_mean),
                           0.0f, thrust::plus<float> ()) / length);
    stats.i_sd = std::sqrt(thrust::transform_reduce(_d_vimag.begin(), _d_vimag.end(), mean_subtraction_square(stats.i_mean),
                           0.0f, thrust::plus<float> ()) / length);
    BOOST_LOG_TRIVIAL(debug) << "DataStatistics r_mean = " << stats.r_mean
                             << " r_sd =  " << stats.r_sd
                             << " i_mean = " << stats.i_mean
                             << " i_sd = " << stats.i_sd;
}

void SKRfiReplacementCuda::generate_replacement_data(const DataStatistics &stats, 
                                                     thrust::device_vector<thrust::complex<float>> &replacement_data)
{
    BOOST_LOG_TRIVIAL(info) << "generating replacement data..\n";
    thrust::host_vector<thrust::complex<float>> h_replacement_data(_window_size);
    replacement_data = h_replacement_data;
    thrust::minstd_rand gen;
    thrust::random::normal_distribution<float> rdist(stats.r_mean, stats.r_sd);
    thrust::random::normal_distribution<float> idist(stats.i_mean, stats.i_sd);
    for(std::size_t index = 0; index < _window_size; index++){
        replacement_data[index] = thrust::complex<float>(rdist(gen), idist(gen));
    }
}

void SKRfiReplacementCuda::replace_rfi_data(const thrust::device_vector<int> &rfi_status,
                                            thrust::device_vector<thrust::complex<float>> &data)
{
    _rfi_status = rfi_status;
    DataStatistics stats;
    thrust::device_vector<thrust::complex<float>> replacement_data;
    //initialize data members of the class
    init();
    //RFI present and not in all windows
    if((_nrfi_windows > 0) && (_nrfi_windows < _nwindows)){
        get_clean_data_statistics(data, stats);
        generate_replacement_data(stats, replacement_data);
	//Replacing RFI
	for(std::size_t ii = 0; ii < _nrfi_windows; ii++){
            std::size_t index = _rfi_window_indices[ii] * _window_size;
	    thrust::copy(replacement_data.begin(), replacement_data.end(), (data.begin() +index));
        }
    }
}

} //edd
} //effelsberg
} //psrdada_cpp
