#include "/src/psrdada_cpp/psrdada_cpp/effelsberg/edd/test/SKTestVector.hpp"

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {
namespace test {

SKTestVector::SKTestVector(int sample_size, int window_size, bool with_rfi)
    : _sample_size(sample_size),
      _window_size(window_size),
      _with_rfi(with_rfi)
{
}

SKTestVector::~SKTestVector()
{
}

void SKTestVector::generate_normal_distribution_vector(std::vector<std::complex<float>> &samples)
{
    std::vector<float> real(_sample_size), imag(_sample_size);
    samples.resize(_sample_size);
    //normal distribution seeds
    std::default_random_engine real_gen(time(NULL));
    std::default_random_engine imag_gen(2);
    //Normal distributions
    std::normal_distribution<float> real_dist(MEAN, STD);
    std::normal_distribution<float> imag_dist(MEAN, STD);
    //Complex vector
    for(int index = 0; index < _sample_size; index++){
        real[index] = real_dist(real_gen);
        imag[index] = imag_dist(imag_gen);
        samples[index] = std::complex<float>(real[index], imag[index]);
    }
}

void SKTestVector::generate_sine_vector(std::vector<std::complex<float>> &sine_vector)
{
    std::vector<float> sine_real(_window_size);
    sine_vector.resize(_window_size);
    for(int t = 0; t < _window_size; t++){
        sine_real[t] = std::sin(2 * 3.14 * (0.25 / _window_size) * t); // number of cycles per window = 0.25.  
        //converting to complex values - real only.
        sine_vector[t] = std::sin(std::complex<float>(sine_real[t],0));
    }
}

void SKTestVector::generate_test_vector(std::vector<int> rfi_window_indices, std::vector<std::complex<float>> &test_vector)
{
    test_vector.resize(_sample_size);
    generate_normal_distribution_vector(test_vector);
    if(_with_rfi){
        std::vector<std::complex<float>> rfi_vector(_window_size);
        generate_sine_vector(rfi_vector);
        int nwindows = rfi_window_indices.size();
        for(int win = 0; win < nwindows; win++){
            int istart = rfi_window_indices[win] * _window_size;
	    int iend = istart + _window_size;
	    BOOST_LOG_TRIVIAL(debug) <<"RFI index = " << rfi_window_indices[win] <<"\n";
            std::transform((test_vector.begin() + istart), (test_vector.begin() + iend), rfi_vector.begin(),
			   (test_vector.begin() + istart), std::plus<std::complex<float>>());
	}
    }
}

} //test
} //edd
} //effelsberg
} //psrdada_cpp

