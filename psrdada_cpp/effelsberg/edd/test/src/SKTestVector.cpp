#include "psrdada_cpp/effelsberg/edd/test/SKTestVector.hpp"

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {
namespace test {

SKTestVector::SKTestVector(std::size_t sample_size, std::size_t window_size, bool with_rfi, float mean, float std)
    : _sample_size(sample_size),
      _window_size(window_size),
      _with_rfi(with_rfi),
      _mean(mean),
      _std(std)
{
    BOOST_LOG_TRIVIAL(debug) << "Creating SKTestVector instance..\n";
}

SKTestVector::~SKTestVector()
{
    BOOST_LOG_TRIVIAL(debug) << "Destroying SKTestVector instance..\n";
}

void SKTestVector::generate_normal_distribution_vector(std::vector<std::complex<float>> &samples)
{
    samples.resize(_sample_size);
    //normal distribution seed
    std::default_random_engine gen(1);
    //generating normal distribution samples
    BOOST_LOG_TRIVIAL(debug) << "generating normal distribution samples for mean: " << _mean 
    << " and standard deviation: " << _std << "\n";
    std::normal_distribution<float> real_dist(_mean, _std);
    std::normal_distribution<float> imag_dist(_mean, _std);
    //Complex vector
    for(std::size_t index = 0; index < _sample_size; index++){
        float real = real_dist(gen);
        float imag = imag_dist(gen);
        samples[index] = std::complex<float>(real, imag);
    }
}

void SKTestVector::generate_sine_vector(std::vector<std::complex<float>> &sine_vector)
{
    sine_vector.resize(_window_size);
    BOOST_LOG_TRIVIAL(debug) << "generating sine samples of size: " << _window_size; 
    for(std::size_t t = 0; t < _window_size; t++){
        float sine_real = std::sin(2 * M_PI * (0.25 / _window_size) * t); // number of cycles per window = 0.25.  
        //converting to complex values - real only.
        sine_vector[t] = std::sin(std::complex<float>(sine_real, 0));
    }
}

void SKTestVector::generate_test_vector(std::vector<int> const& rfi_window_indices, std::vector<std::complex<float>> &test_vector)
{
    test_vector.resize(_sample_size);
    generate_normal_distribution_vector(test_vector);
    BOOST_LOG_TRIVIAL(debug) << "generating test vector" ; 
    if(_with_rfi){
        BOOST_LOG_TRIVIAL(debug) << " with RFI\n"; 
        std::vector<std::complex<float>> rfi_vector(_window_size);
        generate_sine_vector(rfi_vector);
        int nwindows = rfi_window_indices.size();
	BOOST_LOG_TRIVIAL(debug) <<"adding rfi in windows.." << "\n";
        for(int win = 0; win < nwindows; win++){
            int istart = rfi_window_indices[win] * _window_size;
            int iend = istart + _window_size;
	    BOOST_LOG_TRIVIAL(debug) << " " << rfi_window_indices[win];
            std::transform((test_vector.begin() + istart), (test_vector.begin() + iend), rfi_vector.begin(),
			   (test_vector.begin() + istart), std::plus<std::complex<float>>());
	}
    }
}
} //test
} //edd
} //effelsberg
} //psrdada_cpp

