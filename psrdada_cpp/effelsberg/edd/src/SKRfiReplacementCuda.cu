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

struct generate_replacement_data{
    float normal_dist_mean, normal_dist_std;
    generate_replacement_data(float mean, float std) 
        : normal_dist_mean(mean),
          normal_dist_std(std)
    {};
    __host__ __device__
    thrust::complex<float> operator() (unsigned int n) const{
        thrust::minstd_rand gen;
	thrust::random::normal_distribution<float> dist(normal_dist_mean, normal_dist_std);
	gen.discard(n);
	return thrust::complex<float> (dist(gen), dist(gen));
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
    nvtxRangePushA("get_rfi_window_indices");
    _nrfi_windows = thrust::count(_rfi_status.begin(), _rfi_status.end(), 1);
    _rfi_window_indices.resize(_nrfi_windows);
    thrust::copy_if(thrust::make_counting_iterator<int>(0),
                    thrust::make_counting_iterator<int>(_nwindows),
                    _rfi_status.begin(),
                    _rfi_window_indices.begin(),
                    thrust::placeholders::_1 == 1);
    nvtxRangePop();
}

void SKRfiReplacementCuda::get_clean_window_indices()
{
    nvtxRangePushA("get_clean_window_indices");
    _nclean_windows = thrust::count(_rfi_status.begin(), _rfi_status.end(), 0);
    _clean_window_indices.resize(DEFAULT_NUM_CLEAN_WINDOWS);
    thrust::copy_if(thrust::make_counting_iterator<int>(0),
                    thrust::make_counting_iterator<int>(_nwindows),
                    _rfi_status.begin(),
                    _clean_window_indices.begin(),
                    thrust::placeholders::_1 == 0);
    nvtxRangePop();
}

void SKRfiReplacementCuda::get_clean_data_statistics(const thrust::device_vector<thrust::complex<float>> &data)
{
    nvtxRangePushA("get_clean_data_statistics");
    _window_size = data.size() / _nwindows;
    _clean_data.resize(DEFAULT_NUM_CLEAN_WINDOWS * _window_size);
    for(std::size_t ii = 0; ii < DEFAULT_NUM_CLEAN_WINDOWS; ii++){
        std::size_t window_index = _clean_window_indices[ii];
        std::size_t ibegin = window_index * _window_size;
        std::size_t iend = ibegin + _window_size - 1;
        std::size_t jj = ii * _window_size;
        thrust::copy((data.begin() + ibegin), (data.begin() + iend), (_clean_data.begin() + jj));
        BOOST_LOG_TRIVIAL(debug) <<"clean_win_index = " << window_index
                                 << " ibegin = " << ibegin << " iend = " << iend;
    }
    nvtxRangePop();
    compute_clean_data_statistics();
}

void SKRfiReplacementCuda::compute_clean_data_statistics() 
{
    nvtxRangePushA("compute_clean_data_statistics");
    std::size_t length = _clean_data.size();
    _ref_mean = (thrust::reduce(_clean_data.begin(), _clean_data.end(), thrust::complex<float> (0.0f, 0.0f))). real() / length;
    _ref_sd = std::sqrt(thrust::transform_reduce(_clean_data.begin(), _clean_data.end(), mean_subtraction_square(_ref_mean),
                        0.0f, thrust::plus<float> ()) / length);
    nvtxRangePop();
    BOOST_LOG_TRIVIAL(debug) << "DataStatistics mean = " << _ref_mean
                             << " sd =  " << _ref_sd;
}

void SKRfiReplacementCuda::replace_rfi_data(const thrust::device_vector<int> &rfi_status,
                                            thrust::device_vector<thrust::complex<float>> &data)
{
    nvtxRangePushA("replace_rfi_data");
    _rfi_status = rfi_status;
    thrust::device_vector<thrust::complex<float>> replacement_data;
    //initialize data members of the class
    init();
    //RFI present and not in all windows
    if((_nrfi_windows > 0) && (_nrfi_windows < _nwindows)){
        get_clean_data_statistics(data);
	//Replacing RFI
	thrust::counting_iterator<unsigned int> sequence_index_begin(0);
        nvtxRangePushA("replace_rfi_datai_loop");
	for(std::size_t ii = 0; ii < _nrfi_windows; ii++){
            std::size_t index = _rfi_window_indices[ii] * _window_size;
            thrust::transform(sequence_index_begin, (sequence_index_begin + _window_size), 
                              (data.begin() + index), generate_replacement_data(_ref_mean, _ref_sd));
        }
        nvtxRangePop();
    }
    nvtxRangePop();
}
} //edd
} //effelsberg
} //psrdada_cpp
