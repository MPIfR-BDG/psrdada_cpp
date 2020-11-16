#include "psrdada_cpp/effelsberg/edd/SKRfiReplacementCuda.cuh"

#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/random/linear_congruential_engine.h>
#include <nvToolsExt.h>



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


SKRfiReplacementCuda::SKRfiReplacementCuda() {
    BOOST_LOG_TRIVIAL(debug) << "Creating new SKRfiReplacementCuda instance..\n";
}


SKRfiReplacementCuda::~SKRfiReplacementCuda() {
    BOOST_LOG_TRIVIAL(debug) << "Destroying SKRfiReplacementCuda instance..\n";
}


void SKRfiReplacementCuda::replace_rfi_data(const thrust::device_vector<int> &rfi_status,
                                            thrust::device_vector<thrust::complex<float>> &data,
                                            std::size_t clean_windows, cudaStream_t stream) {
    nvtxRangePushA("replace_rfi_data");
    thrust::cuda::par.on(stream);
    thrust::device_vector<thrust::complex<float>> replacement_data;
    //initialize data members of the class

    BOOST_LOG_TRIVIAL(debug) << "getting RFI window indices..\n";
    _rfi_window_indices.resize(thrust::count(rfi_status.begin(), rfi_status.end(), 1));

    thrust::copy_if(thrust::make_counting_iterator<int>(0),
                    thrust::make_counting_iterator<int>(rfi_status.size()),
                    rfi_status.begin(),
                    _rfi_window_indices.begin(),
                    thrust::placeholders::_1 == 1);

    BOOST_LOG_TRIVIAL(debug) << "getting clean window indices..\n";
    size_t _nclean_windows = thrust::count(rfi_status.begin(), rfi_status.end(), 0);
    _clean_window_indices.resize(_nclean_windows);
    thrust::copy_if(thrust::make_counting_iterator<int>(0),
                    thrust::make_counting_iterator<int>(rfi_status.size()),
                    rfi_status.begin(),
                    _clean_window_indices.begin(),
                    thrust::placeholders::_1 == 0);

    if(_nclean_windows < rfi_status.size()){
        //RFI present and not in all windows
        if (_nclean_windows < clean_windows) {
            clean_windows = _nclean_windows;
        }

        BOOST_LOG_TRIVIAL(debug) << "collecting clean data from chosen number of clean windows..\n";
        std::size_t _window_size = data.size() / rfi_status.size();
        _clean_data.resize(clean_windows * _window_size);
        for(std::size_t ii = 0; ii < clean_windows; ii++){
            std::size_t window_index = _clean_window_indices[ii];
            std::size_t ibegin = window_index * _window_size;
            std::size_t iend = ibegin + _window_size - 1;
            std::size_t jj = ii * _window_size;
            thrust::copy((data.begin() + ibegin), (data.begin() + iend), (_clean_data.begin() + jj));
            BOOST_LOG_TRIVIAL(debug) <<"clean_win_index = " << window_index
                                     << " ibegin = " << ibegin << " iend = " << iend;
        }

        BOOST_LOG_TRIVIAL(debug) << "computing statistics of clean data..\n";
        //The distribution of both real and imag are expected to ahve  same mean and standard deviation.
        //Therefore computing _ref_mean, _ref_sd for real distribution only.
        std::size_t length = _clean_data.size();
        float _ref_mean = (thrust::reduce(_clean_data.begin(), _clean_data.end(), thrust::complex<float> (0.0f, 0.0f))).  real() / length;
        float _ref_sd = std::sqrt(thrust::transform_reduce(_clean_data.begin(), _clean_data.end(), mean_subtraction_square(_ref_mean),
                            0.0f, thrust::plus<float> ()) / length);
        BOOST_LOG_TRIVIAL(debug) << "DataStatistics mean = " << _ref_mean
                                 << " sd =  " << _ref_sd;
        //Replacing RFI
        thrust::counting_iterator<unsigned int> sequence_index_begin(0);
        nvtxRangePushA("replace_rfi_datai_loop");
        for(std::size_t ii = 0; ii < _rfi_window_indices.size(); ii++){
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
