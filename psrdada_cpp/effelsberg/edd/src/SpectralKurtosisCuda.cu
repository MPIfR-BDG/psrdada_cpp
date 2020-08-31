#include "psrdada_cpp/effelsberg/edd/SpectralKurtosisCuda.cuh"

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {

struct compute_power{
    __host__ __device__
    float operator()(thrust::complex<float> z)
    {
        return (thrust::abs(z) * thrust::abs(z));
    }
};

struct power_square{
    __host__ __device__
    float operator()(thrust::complex<float> z)
    {
        float abs = thrust::abs(z);
        float power = abs * abs;
        return (power * power);
    }
};

struct check_rfi{
    const std::size_t M; //_window_size
    const float sk_min;
    const float sk_max;
    check_rfi(std::size_t m, float min, float max)
        : M(m),
          sk_min(min),
          sk_max(max)
    {}

    __host__ __device__
    float operator() (float s1, float s2) const {
        float sk = ((M + 1) / (M - 1)) * (((M * s2) / (s1 * s1)) - 1);
        return ((sk < sk_min) || (sk > sk_max)) ;
   }
};

__global__ void compute_sk_kernel(thrust::complex<float> *data, std::size_t sample_size, std::size_t window_size,
                                  float sk_max, float sk_min, int *rfi_status)
{
    extern __shared__ float buffer[];
    float *s1 = &buffer[0];
    float *s2 = &buffer[blockDim.x];
    int g_index = threadIdx.x + blockIdx.x * blockDim.x;
    int l_index = threadIdx.x;

    if(l_index < blockDim.x){
        s1[l_index] = thrust::abs(data[g_index]) * thrust::abs(data[g_index]);
        s2[l_index] = s1[l_index] * s1[l_index];
    }
    __syncthreads();

    for(int s = 1; s < blockDim.x; s *= 2){
        int index = 2 * s * l_index;
	if(index < blockDim.x){
            s1[index] += s1[index + s];
            s2[index] += s2[index + s];
	}
	__syncthreads();
    }

    float sk;
    if(l_index == 0){
        sk = ((window_size + 1) / (window_size - 1)) *(((window_size * s2[0]) / (s1[0] * s1[0])) - 1);
        rfi_status[blockIdx.x] = (int) ((sk < sk_min) || (sk > sk_max));
    }
}

SpectralKurtosisCuda::SpectralKurtosisCuda(std::size_t nchannels, std::size_t window_size, float sk_min, float sk_max)
    : _nchannels(nchannels),
      _window_size(window_size),
      _sk_min(sk_min),
      _sk_max(sk_max)
{
    BOOST_LOG_TRIVIAL(debug) << "Creating new SpectralKurtosisCuda instance... \n";
}

SpectralKurtosisCuda::~SpectralKurtosisCuda()
{
    BOOST_LOG_TRIVIAL(debug) << "Destroying SpectralKurtosisCuda instance... \n";
}

void SpectralKurtosisCuda::init()
{
    if((_sample_size % _window_size) != 0){
        BOOST_LOG_TRIVIAL(error) << "Sample(data) size " << _sample_size <<" is not a multiple of _window_size "
                                 << _window_size <<". Give different window size.\n";
        throw std::runtime_error("Data(sample) size is not a multiple of window_size. Give different window size. \n");
    }
    _nwindows = _sample_size /_window_size;
    _d_s1.resize(_nwindows);
    _d_s2.resize(_nwindows);
}

void SpectralKurtosisCuda::compute_sk(const thrust::device_vector<thrust::complex<float>> &data, RFIStatistics &stats){
    nvtxRangePushA("compute_sk");
    _sample_size = data.size();
    BOOST_LOG_TRIVIAL(debug) << "Computing SK (thrust version) for sample_size " << _sample_size
                             << " and window_size " << _window_size <<".\n";
    //initializing class variables
    init();
    //computing _d_s1 for all windows
    thrust::reduce_by_key(thrust::device, 
                          thrust::make_transform_iterator(thrust::counting_iterator<int> (0), (thrust::placeholders::_1 / _window_size)),
                          thrust::make_transform_iterator(thrust::counting_iterator<int> (_sample_size - 1), (thrust::placeholders::_1 / _window_size)), 
                          thrust::make_transform_iterator(data.begin(), compute_power()), 
                          thrust::discard_iterator<int>(), 
                          _d_s1.begin());
    //computing _d_s2  for all windows
    thrust::reduce_by_key(thrust::device, 
                          thrust::make_transform_iterator(thrust::counting_iterator<int> (0), (thrust::placeholders::_1 / _window_size)),
                          thrust::make_transform_iterator(thrust::counting_iterator<int> (_sample_size - 1), (thrust::placeholders::_1 / _window_size)), 
                          thrust::make_transform_iterator(data.begin(), power_square()), 
                          thrust::discard_iterator<int>(), 
                          _d_s2.begin());
    //computes SK and checks the threshold to detect RFI.
    stats.rfi_status.resize(_nwindows);
    thrust::transform(_d_s1.begin(), _d_s1.end(), _d_s2.begin(), stats.rfi_status.begin(), check_rfi(_window_size, _sk_min, _sk_max));
    stats.rfi_fraction = thrust::reduce(stats.rfi_status.begin(), stats.rfi_status.end(), 0.0f) / _nwindows;
    BOOST_LOG_TRIVIAL(info) << "RFI fraction: " << stats.rfi_fraction;
    nvtxRangePop();
}

void SpectralKurtosisCuda::compute_sk_k(thrust::device_vector<thrust::complex<float>> &data, RFIStatistics &stats){
    nvtxRangePushA("compute_sk_kernel");
    _sample_size = data.size();
    BOOST_LOG_TRIVIAL(debug) << "Computing SK (kernel version) for sample_size " << _sample_size
                             << " and window_size " << _window_size <<".\n";

    _nwindows = _sample_size / _window_size;
    stats.rfi_status.resize(_nwindows);
    thrust::complex<float> *k_data = thrust::raw_pointer_cast(data.data());
    int *k_rfi_status = thrust::raw_pointer_cast(stats.rfi_status.data());
    int blockSize = _window_size;
    int gridSize = _sample_size / blockSize;
    int sh_mem_size = 2 * blockSize * sizeof(float); 
    compute_sk_kernel<<<gridSize, blockSize, sh_mem_size>>> (k_data, _sample_size, _window_size, _sk_max, _sk_min, k_rfi_status);
    stats.rfi_fraction = thrust::reduce(stats.rfi_status.begin(), stats.rfi_status.end(), 0.0f) / _nwindows;
    BOOST_LOG_TRIVIAL(info) << "RFI fraction: " << stats.rfi_fraction;
    nvtxRangePop();
}

} //edd
} //effelsberg
} //psrdada_cpp
