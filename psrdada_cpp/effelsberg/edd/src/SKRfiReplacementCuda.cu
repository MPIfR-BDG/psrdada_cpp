#include "psrdada_cpp/effelsberg/edd/SKRfiReplacementCuda.cuh"

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {

struct mean_subtraction_square{
    const float mean;
    mean_subtraction_square(float _mean) :mean(_mean) {}
    __host__ __device__
    float operator() (thrust::complex<float> val) const{
        float x = val.real() - mean;
        return (x * x);
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

void SKRfiReplacementCuda::get_clean_data_statistics(const thrust::device_vector<thrust::complex<float>> &data)
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
    compute_data_statistics(clean_data);
}

void SKRfiReplacementCuda::compute_data_statistics(const thrust::device_vector<thrust::complex<float>> &data) 
{
    std::size_t length = data.size();
    _ref_mean = (thrust::reduce(data.begin(), data.end(), thrust::complex<float> (0.0f, 0.0f))). real() / length;
    _ref_sd = std::sqrt(thrust::transform_reduce(data.begin(), data.end(), mean_subtraction_square(_ref_mean),
                        0.0f, thrust::plus<float> ()) / length);
    BOOST_LOG_TRIVIAL(debug) << "DataStatistics mean = " << _ref_mean
                             << " sd =  " << _ref_sd;
}

void SKRfiReplacementCuda::generate_replacement_data(thrust::device_vector<thrust::complex<float>> &replacement_data)
{
    BOOST_LOG_TRIVIAL(info) << "generating replacement data..\n";
    replacement_data.resize(_window_size);
    thrust::minstd_rand gen;
    thrust::random::normal_distribution<float> ndist(_ref_mean, _ref_sd);
    for(std::size_t index = 0; index < _window_size; index++){
        replacement_data[index] = thrust::complex<float>(ndist(gen), ndist(gen));
    }
}

void SKRfiReplacementCuda::replace_rfi_data(const thrust::device_vector<int> &rfi_status,
                                            thrust::device_vector<thrust::complex<float>> &data)
{
    _rfi_status = rfi_status;
    thrust::device_vector<thrust::complex<float>> replacement_data;
    //initialize data members of the class
    init();
    //RFI present and not in all windows
    if((_nrfi_windows > 0) && (_nrfi_windows < _nwindows)){
        get_clean_data_statistics(data);
        generate_replacement_data(replacement_data);
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
