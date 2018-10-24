//
// Do the input transpose to take the data from SPEAD2/MKRECV
// capture order to beanfarmer order
//
// Inputs:
//  - 8-bit volatages: TAFTP order
//
//  Outputs:
//  - 8-bit voltages: FTPA order
//
//  Notes:
//  - Inner 2 dimensions on the input are always (256, 2)
//  - Inner F dimension is always a multiple of 16
//  - Outer F dimension can be any number
//  - A dimension can be any number between 4 and 64 in steps of 4 (may not all be valid).
//  - Outer T dimension can be any number
//
// Plan:
// - Load in inner T and P dimension (512 threads) for N antennas. This will use
//   256 * 2 * 2 * N bytes, so 32 antennas = 32k bytes (half shared usage), could
//   also do this in groups of 32 time samples (each block would then do this 8
//   times).
// - Read back from shared memory into output order, with shared memory padding
//   if necessary.
// - Performance is probably best for char4 loads

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <cuda.h>
#include <cuComplex.h>
#include <vector>
#include <random>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
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

#define NSAMPS_PER_TIMESTAMP 256
#define NPOL 2
#define MAX_ANTENNAS 32

//Split transpose to convert from TAFTP order to FTPA order


__global__
void split_transpose_k(
    char2 const * __restrict__ input,
    char2 * __restrict__ output,
    int total_nantennas,
    int used_nantennas,
    int start_antenna,
    int nchans,
    int ntimestamps)
{
    __shared__ char2 transpose_buffer[MAX_ANTENNAS][NSAMPS_PER_TIMESTAMP][NPOL];

    //TAFTP (input dimensions)
    const int tp = NSAMPS_PER_TIMESTAMP * NPOL;
    const int ftp = nchans * tp;
    const int aftp = total_nantennas * ftp;

    //FTPA
    const int pa = NPOL * used_nantennas;
    const int tpa = ntimestamps * NSAMPS_PER_TIMESTAMP * pa;


    int nantennas_sets = ceilf(((float) used_nantennas) / MAX_ANTENNAS);

    for (int timestamp_idx = blockIdx.x; timestamp_idx < ntimestamps; timestamp_idx += gridDim.x)
    {
        for (int chan_idx = blockIdx.y; chan_idx < nchans; chan_idx += gridDim.y)
        {
            for (int antenna_set_idx = 0; antenna_set_idx < nantennas_sets; ++antenna_set_idx)
            {
                int remaining_antennas = min(used_nantennas - antenna_set_idx * MAX_ANTENNAS, MAX_ANTENNAS);
                // Load data into shared memory
                for (int antenna_idx = threadIdx.y; antenna_idx < remaining_antennas; antenna_idx += blockDim.y)
                {
                    int input_antenna_idx = antenna_set_idx * MAX_ANTENNAS + antenna_idx + start_antenna;

                    for (int samppol_idx = threadIdx.x; samppol_idx < (NSAMPS_PER_TIMESTAMP * NPOL); samppol_idx += blockDim.x)
                    {
                        int pol_idx = samppol_idx%NPOL;
                        int samp_idx = samppol_idx/NPOL;
                        int input_idx = timestamp_idx * aftp + input_antenna_idx * ftp + chan_idx * tp + samp_idx * NPOL + pol_idx;
                        transpose_buffer[antenna_idx][samp_idx][pol_idx] = input[input_idx];
                    }
                }

                __syncthreads();

                for (int pol_idx = 0; pol_idx < NPOL; ++pol_idx)
                {
                    for (int samp_idx = threadIdx.y; samp_idx < NSAMPS_PER_TIMESTAMP; samp_idx += blockDim.y)
                    {
                        int output_sample_idx = samp_idx + timestamp_idx * NSAMPS_PER_TIMESTAMP;
                        for (int antenna_idx = threadIdx.x; antenna_idx < remaining_antennas; antenna_idx += blockDim.x)
                        {
                            int output_antenna_idx = antenna_set_idx * MAX_ANTENNAS + antenna_idx;

                            //FTPA
                            int output_idx = chan_idx * tpa + output_sample_idx * pa + pol_idx * used_nantennas + output_antenna_idx;

                            output[output_idx] = transpose_buffer[antenna_idx][samp_idx][pol_idx];
                        }
                    }
                }

                __syncthreads();
            }
        }
    }
}


__global__
void input_transpose_simple_k(
    char2 const * __restrict__ input,
    char2 * __restrict__ output,
    int nantennas,
    int nchans,
    int ntimestamps)
{
    //TAFTP
    const int tp = NSAMPS_PER_TIMESTAMP * NPOL;
    const int ftp = nchans * tp;
    const int aftp = nantennas * ftp;

    //FTPA
    const int pa = NPOL * nantennas;
    const int tpa = ntimestamps * NSAMPS_PER_TIMESTAMP * pa;


    int nantennas_sets = ceilf(((float) nantennas) / MAX_ANTENNAS);

    for (int timestamp_idx = blockIdx.x; timestamp_idx < ntimestamps; timestamp_idx += gridDim.x)
    {
        //int input_timestamp_offset = timestamp_idx * input_per_timestamp_offset;

        for (int chan_idx = blockIdx.y; chan_idx < nchans; chan_idx += gridDim.y)
        {
            //int input_chan_offset = input_timestamp_offset + chan_idx * input_per_chan_offset;
            for (int antenna_idx = threadIdx.y; antenna_idx < nantennas; antenna_idx += blockDim.y)
            {
                for (int samppol_idx = threadIdx.x; samppol_idx < (NSAMPS_PER_TIMESTAMP * NPOL); samppol_idx += blockDim.x)
                {
                    int pol_idx = samppol_idx%NPOL;
                    int samp_idx = samppol_idx/NPOL;
                    int input_idx = timestamp_idx * aftp + antenna_idx * ftp + chan_idx * tp + samp_idx * NPOL + pol_idx;
                    int output_sample_idx = samp_idx + timestamp_idx * NSAMPS_PER_TIMESTAMP;
                    int output_idx = chan_idx * tpa + output_sample_idx * pa + pol_idx * nantennas + antenna_idx;
                    //printf("(%d, %d, %d), (%d, %d, %d ), ant=(%d, %d), samp=%d, pol=%d\n",
                    //    blockIdx.x, blockIdx.y, blockIdx.z,
                    //    threadIdx.x, threadIdx.y, threadIdx.z,
                    //    antenna_idx, input_antenna_idx, samp_idx, pol_idx);
                    output[output_idx] = input[input_idx];
                }
            }
        }
    }
}


void c_reference(
    char2 const * __restrict__ input,
    char2 * __restrict__ output,
    int total_nantennas,
    int used_nantennas,
    int start_antenna,
    int nchans,
    int ntimestamps)
{

    //TAFTP to FTPA
    //Input dimensions
    int tp = NSAMPS_PER_TIMESTAMP * NPOL;
    int ftp = nchans * tp;
    int aftp = total_nantennas * ftp;

    //Output dimensions
    int pa = NPOL * used_nantennas;
    int tpa = NSAMPS_PER_TIMESTAMP * ntimestamps * pa;

    for (int timestamp_idx = 0; timestamp_idx < ntimestamps; ++timestamp_idx)
    {
        for (int antenna_idx = 0; antenna_idx < used_nantennas; ++antenna_idx)
        {
            int input_antenna_idx = antenna_idx + start_antenna;
            for (int chan_idx = 0; chan_idx < nchans; ++chan_idx)
            {
                for (int samp_idx = 0; samp_idx < NSAMPS_PER_TIMESTAMP; ++samp_idx)
                {
                    for (int pol_idx = 0; pol_idx < NPOL; ++pol_idx)
                    {
                        int input_idx = timestamp_idx * aftp + input_antenna_idx * ftp + chan_idx * tp + samp_idx * NPOL + pol_idx;
                        int output_sample_idx = timestamp_idx * NSAMPS_PER_TIMESTAMP + samp_idx;
                        int output_idx = chan_idx * tpa + output_sample_idx * pa + pol_idx * used_nantennas + antenna_idx;
                        output[output_idx] = input[input_idx];
                    }
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

template <class T>
void dump_host_buffer(T* buffer, size_t size, std::string filename)
{
    std::ofstream infile;
    infile.open(filename.c_str(),std::ifstream::out | std::ifstream::binary);
    infile.write((char*)buffer ,size*sizeof(T));
    infile.close();
}

int main()
{

    int total_nantennas = 64;
    int nchans = 64;
    int ntimestamps = 32;
    int used_nantennas = 15;
    int start_antenna = 7;
    int niterations = 100;

    std::size_t input_size = ntimestamps * total_nantennas * nchans * NSAMPS_PER_TIMESTAMP * NPOL;
    std::size_t output_size = ntimestamps * used_nantennas * nchans * NSAMPS_PER_TIMESTAMP * NPOL;
    CUDA_ERROR_CHECK(cudaSetDevice(0));
    CUDA_ERROR_CHECK(cudaDeviceReset());

    /**
     * Below we set default values for the arrays. Beamforming this data should result in
     * every output having the same value.
     *
     */
#ifdef TEST_CORRECTNESS

    std::cout << "Generating host test vectors...\n";
    char2 default_value = {0, 0};
    thrust::host_vector<char2> input_h(input_size, default_value);
    thrust::host_vector<char2> output_h(output_size, default_value);

    populate<char2>(input_h.data(), input_size, 0, 64);

    /*
    for (int ii=0; ii<size; ++ii)
    {
        input_h[ii].x = (ii%256)-127;
    input_h[ii].y = ((ii/256)%256 - 127);
    }
    */

    dump_host_buffer<char2>(thrust::raw_pointer_cast(input_h.data()), input_size, "input_data.bin");


    thrust::device_vector<char2> input = input_h;
    thrust::device_vector<char2> output = output_h;
#else
    std::cout << "NOT generating host test vectors...\n";
    thrust::device_vector<char2> input(input_size);
    thrust::device_vector<char2> output(output_size);
#endif //TEST_CORRECTNESS

    char2* input_ptr = thrust::raw_pointer_cast(input.data());
    char2* output_ptr = thrust::raw_pointer_cast(output.data());

    dim3 grid(ntimestamps, nchans, 1);
    dim3 block(512, 1, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout << "Executing warm up\n";
    //Warm up
    for (int jj=0; jj<niterations; ++jj)
    {
        split_transpose_k<<< grid, block >>>( input_ptr, output_ptr, total_nantennas, used_nantennas, start_antenna, nchans, ntimestamps);
    }
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    std::cout << "Starting benchmarking\n";
    cudaEventRecord(start);
    for (int ii=0; ii<niterations; ++ii)
    {
      split_transpose_k<<< grid, block >>>( input_ptr, output_ptr, total_nantennas, used_nantennas, start_antenna, nchans, ntimestamps);
    }
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Total kernel duration (ms): " << milliseconds << "\n";

#ifdef TEST_CORRECTNESS
    std::cout << "Testing correctness...\n";
    thrust::host_vector<char2> gpu_output = output;
    thrust::host_vector<char2> cpu_output(output_size);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    c_reference(input_h.data(), cpu_output.data(), total_nantennas, used_nantennas, start_antenna, nchans, ntimestamps);

    dump_host_buffer<char2>(thrust::raw_pointer_cast(gpu_output.data()), output_size, "gpu_output_data.bin");
    dump_host_buffer<char2>(thrust::raw_pointer_cast(cpu_output.data()), output_size, "cpu_output_data.bin");

if (!is_same(cpu_output.data(), gpu_output.data(), output_size))
        std::cout << "FAILED!\n";
    else
        std::cout << "PASSED!\n";
#endif
}