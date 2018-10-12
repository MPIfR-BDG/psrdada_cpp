// inputs:
//  - delays in array format: beam, antenna, rate, offset
//  - channel centre frequencies
//
//  outputs:
//  - weights in 8-bit (for multiple epochs?):
//  timeset, frequency, beam, antenna order
//
// Weights are simply:
//
// delay = unix time * delay rate + delay offset
// phase = delay * channel frequency
// weight = exp(i * phase)
//
// For speed the weight can be calculated with the fast
// sine and cosine approximations using:
//
// weight = sincos(2*pi*angle) //Need to check the 2 pi bit
//
// The magnitude of the weight here is guaranteed to be 1
// these should be rescaled to signed 8-bit ints. The easiest
// way is simply to multiply by 127 and round nearest before
// downcasting to int8_t (should pack these into char4 for
// read/write efficiency)

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <cuda.h>
#include <cuComplex.h>
#include <vector>
#include <random>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <math.h>
#include <cmath>

#define TEST_CORRECTNESS 0

#define TWOPI 6.283185307179586f
#define CUDA_ERROR_CHECK(ans) { cuda_assert_success((ans), __FILE__, __LINE__); }

/**
 * @brief Function that raises an error on receipt of any cudaError_t
 *  value that is not cudaSuccess
 */
inline void cuda_assert_success(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        std::stringstream error_msg;
        error_msg << "CUDA failed with error: "
              << cudaGetErrorString(code) << std::endl
              << "File: " << file << std::endl
              << "Line: " << line << std::endl;
        throw std::runtime_error(error_msg.str());
    }
}

__global__
void generate_weights_k(
    float2 const * __restrict__ delay_models,
    char2 * __restrict__ weights,
    float const * __restrict__ channel_frequencies,
    int nantennas,
    int nbeams,
    int nchans,
    float tstart,
    float tstep,
    int ntsteps)
{

    //for each loaded delay poly we can produce multiple epochs for one antenna, one beam, all frequencies and both pols
    //Different blocks should handle different beams (as antennas are on the inner dimension of the output product)

    //Basics of this kernel:
    //
    // gridDim.x is used for beams (there is a loop if you want to limit the grid size)
    // gridDim.y is used for channels (there is a loop if you want to limit the grid size)
    // blockDim.x is used for antennas (there is a loop if you want to limit the grid size)
    //
    // Time steps are handled in a the inner loop. As antennas are on the inner dimension of
    // both the input and the output array, all reads and writes should be coalesced.
    const int weights_per_beam = nantennas;
    const int weights_per_channel = weights_per_beam * nbeams;
    const int weights_per_time_step = weights_per_channel * nchans;

    float2 weight;
    char2 compressed_weight;
    //This isn't really needed as there will never be more than 64 antennas
    //However this makes this fucntion more flexible with smaller blocks

    for (int chan_idx = blockIdx.y; chan_idx < nchans; chan_idx += gridDim.y)
    {
        float frequency = channel_frequencies[chan_idx];
        int chan_offset = chan_idx * weights_per_channel; // correct

        for (int beam_idx = blockIdx.x; beam_idx < nbeams; beam_idx += gridDim.x)
        {
            int beam_offset = chan_offset + beam_idx * weights_per_beam; // correct

            for (int antenna_idx = threadIdx.x; antenna_idx < nantennas; antenna_idx+=blockDim.x)
            {
                float2 delay_model = delay_models[beam_idx * nantennas + antenna_idx]; // correct

                int antenna_offset = beam_offset + antenna_idx;

                for (int time_idx = threadIdx.y; time_idx < ntsteps; time_idx+=blockDim.y)
                {
                    //Calculates epoch offset
                    float t = tstart + time_idx * tstep;
                    float phase = (t * delay_model.x + delay_model.y) * frequency;
                    //This is possible as the magnitude of the weight is 1
                    //If we ever have to implement scalar weightings, this
                    //must change.
                    __sincosf(TWOPI * phase, &weight.y, &weight.x);
                    compressed_weight.x = (char) __float2int_rn(weight.x * 127.0f);
                    compressed_weight.y = (char) __float2int_rn(weight.y * 127.0f);
                    int output_idx = time_idx * weights_per_time_step + antenna_offset;
                    weights[output_idx] = compressed_weight;
                }
            }
        }

    }
}

void c_reference(
    float2 const * delay_models,
    char2 * weights,
    float  const * channel_frequencies,
    int nantennas,
    int nbeams,
    int nchans,
    float tstart,
    float tstep,
    int ntsteps)
{
    float2 weight;
    char2 compressed_weight;
    for (int antenna_idx = 0; antenna_idx < nantennas; ++antenna_idx)
    {
        for (int beam_idx = 0; beam_idx < nbeams; ++beam_idx)
        {

            float2 delay_model = delay_models[beam_idx * nantennas + antenna_idx];

            for (int chan_idx = 0; chan_idx < nchans; ++chan_idx)
            {

                float frequency = channel_frequencies[chan_idx];

                for (int time_idx = 0; time_idx < ntsteps; ++time_idx)
                {
                    float t = tstart + time_idx * tstep;
                    float phase = (t * delay_model.x + delay_model.y) * frequency;
                    sincosf(TWOPI * phase, &weight.y, &weight.x);
                    compressed_weight.x = (char) round(weight.x * 127.0f);
                    compressed_weight.y = (char) round(weight.y * 127.0f);
                    int output_idx = nantennas * ( nbeams * ( time_idx * nchans + chan_idx ) + beam_idx ) + antenna_idx;
                    weights[output_idx] = compressed_weight;
                }
            }
        }
    }
}


bool is_same(char2 const * a, char2 const * b, std::size_t size)
{

    int nfail = 0;

    for (std::size_t idx = 0; idx < size; ++idx)
    {


        char2 tmpa = a[idx];
        char2 tmpb = b[idx];

        if (  !((abs(tmpa.x - tmpb.x)<=1) && (abs(tmpa.y - tmpb.y)<=1))  )
        {
            std::cout << "Expected " << (int) tmpa.x << ", " << (int) tmpa.y << " got " << (int) tmpb.x << ", " << (int) tmpb.y << "\n";
            ++nfail;
            if (nfail > 10)
                return false;
        }
    }
    return true;
}

template <typename ComplexType>
void populate(ComplexType* data, std::size_t size, float mean, float std)
{
  std::random_device rd;
  std::mt19937 eng(rd());
  std::normal_distribution<float> distr(mean, std);
  for(std::size_t n = 0 ; n < size; ++n)
    {
      data[n].x = distr(eng);
      data[n].y = distr(eng);
    }
}

int main()
{

    int nantennas = 64;
    int nbeams = 1024;
    int nchans = 64;
    int ntsteps = 100;
    float tstep = 0.001f;
    float fbottom = 1420.0e9;
    float cbw = 265e6;
    float tstart = 0.0f;
    int niterations = 100;


    std::size_t baro_delays_size = nantennas * nbeams;
    std::size_t fba_weights_size = nantennas * nbeams * nchans * ntsteps;

    CUDA_ERROR_CHECK(cudaSetDevice(0));
    CUDA_ERROR_CHECK(cudaDeviceReset());

    /**
     * Below we set default values for the arrays. Beamforming this data should result in
     * every output having the same value.
     *
     */
#ifdef TEST_CORRECTNESS

    std::cout << "Generating host test vectors...\n";
    float2 f2_default_value = {0.0f,0.0f};
    char2 c2_default_value = {0, 0};
    thrust::host_vector<float2> delay_vector_h(baro_delays_size, f2_default_value);
    thrust::host_vector<char2> weights_vector_h(fba_weights_size, c2_default_value);
    thrust::host_vector<float> frequencies_vector_h(nchans, 0.0f);
    for (int ii = 0; ii < nchans; ++ii)
    {
        frequencies_vector_h[ii] = fbottom + cbw * ii;
    }
    populate<float2>(delay_vector_h.data(), baro_delays_size, 1e-11f, 1e-10f);
    thrust::device_vector<float2> delay_vector = delay_vector_h;
    thrust::device_vector<char2> weights_vector = weights_vector_h;
    thrust::device_vector<float> frequencies_vector = frequencies_vector_h;
#else
    std::cout << "NOT generating host test vectors...\n";
    thrust::device_vector<float2> delay_vector(baro_delays_size);
    thrust::device_vector<char2> weights_vector(fba_weights_size);
    thrust::device_vector<float> frequencies_vector(nchans);
#endif //TEST_CORRECTNESS

    float2 const* delays = thrust::raw_pointer_cast(delay_vector.data());
    char2* weights = thrust::raw_pointer_cast(weights_vector.data());
    float const* frequencies = thrust::raw_pointer_cast(frequencies_vector.data());
    dim3 grid(nbeams, nchans, 1);
    dim3 block(32, 32, 1);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout << "Executing warm up\n";
    //Warm up
    for (int jj=0; jj<niterations; ++jj)
    {
        generate_weights_k<<< grid, block >>>( delays, weights, frequencies, nantennas, nbeams, nchans, tstart, tstep, ntsteps);
    }
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    std::cout << "Starting benchmarking\n";
    cudaEventRecord(start);
    for (int ii=0; ii<niterations; ++ii)
    {
      generate_weights_k<<< grid, block >>>( delays, weights, frequencies, nantennas, nbeams, nchans, tstart, tstep, ntsteps);
    }
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Total kernel duration (ms): " << milliseconds << "\n";

#ifdef TEST_CORRECTNESS
    std::cout << "Testing correctness...\n";
    thrust::host_vector<char2> gpu_output = weights_vector;
    thrust::host_vector<char2> cpu_output(fba_weights_size);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    c_reference(delay_vector_h.data(), cpu_output.data(), frequencies_vector_h.data(), nantennas, nbeams, nchans, tstart, tstep, ntsteps);
    if (!is_same(cpu_output.data(), gpu_output.data(), fba_weights_size))
        std::cout << "FAILED!\n";
    else
        std::cout << "PASSED!\n";
#endif
}