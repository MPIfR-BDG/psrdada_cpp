#include "psrdada_cpp/effelsberg/edd/ScaledTransposeTFtoTFT.cuh"
#include "psrdada_cpp/cuda_utils.hpp"

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {
namespace kernels {

__global__
void tf_to_tft_transpose(
    float2 const* __restrict__ input, 
    char2* __restrict__ output, 
    const int nchans, 
    const int nsamps, 
    const int nsamps_per_packet,
    const int nsamps_per_load,
    const float scale,
    const float offset)
{
    extern __shared__ char2 temp[]; //nbytes = sizeof(char2) * nsamps_per_load * nchans;
    const int load_offset = nsamps_per_packet * blockIdx.x * nchans;
    for (int sub_samp_load_idx = 0;
        sub_samp_load_idx < nsamps_per_packet/nsamps_per_load;
        ++sub_samp_load_idx)
    {
        for (int samp_load_idx = threadIdx.y;
            samp_load_idx < nsamps_per_load;
            samp_load_idx += blockDim.y)
        {
            for (int chan_load_idx = threadIdx.x; 
                chan_load_idx < nchans;
                chan_load_idx += blockDim.x)
            {
                int load_idx = (load_offset + (sub_samp_load_idx * nsamps_per_load 
                    + samp_load_idx) * nchans + chan_load_idx);
                float2 val = input[load_idx];
                char2 store_val;
                store_val.x = (char)((val.x - offset)/scale);
                store_val.y = (char)((val.y - offset)/scale);
                temp[samp_load_idx * nsamps_per_load + chan_load_idx] = store_val;
            }
        }
        __syncthreads();
        for (int chan_store_idx = threadIdx.y;
            chan_store_idx < nchans;
            chan_store_idx += blockDim.y)
        {
            for (int samp_store_idx = threadIdx.x;
                samp_store_idx < nsamps_per_load;
                samp_store_idx += blockIdx.x)
            {
                int store_idx = (load_offset + chan_store_idx * nsamps_per_packet 
                    + nsamps_per_load * sub_samp_load_idx + samp_store_idx);
                output[store_idx] = temp[samp_store_idx * nsamps_per_load + chan_store_idx];
            }
        }
        __syncthreads();
    }
}

} //namespace kernels

ScaledTransposeTFtoTFT::ScaledTransposeTFtoTFT(
    int nchans, int nsamps_per_packet, 
    float scale, float offset, cudaStream_t stream)
    : _nchans(nchans)
    , _nsamps_per_packet(nsamps_per_packet)
    , _scale(scale)
    , _offset(offset)
    , _stream(stream)
{

}

ScaledTransposeTFtoTFT::~ScaledTransposeTFtoTFT()
{

}

void ScaledTransposeTFtoTFT::transpose(InputType const& input, OutputType& output)
{
    BOOST_LOG_TRIVIAL(debug) << "Preparing scaled transpose";
    const int max_threads = 1024;
    const int dim_x = std::min(_nchans, max_threads);
    const int dim_y = max_threads/dim_x;
    //assert sizes
    assert(input.size() % (_nchans * _nsamps_per_packet) == 0 /* Input is not a multiple of _nchans * _nsamps_per_packet*/);
    output.resize(input.size());
    const int nsamps_per_load = 16;
    assert((_nsamps_per_packet % nsamps_per_load) == 0);
    const int nsamps = input.size() / _nchans;   
    int shared_mem_bytes = sizeof(OutputType::value_type) * _nchans * nsamps_per_load;
    int nblocks = nsamps / _nsamps_per_packet;
    BOOST_LOG_TRIVIAL(debug) << "Scaled transpose will use " << shared_mem_bytes << " bytes of shared memory.";
    dim3 grid(nblocks);
    dim3 block(dim_x, dim_y);
    InputType::value_type const* input_ptr = thrust::raw_pointer_cast(input.data());
    OutputType::value_type* output_ptr = thrust::raw_pointer_cast(output.data());
    BOOST_LOG_TRIVIAL(debug) << "Executing scaled transpose";
    kernels::tf_to_tft_transpose<<<grid, block, shared_mem_bytes, _stream>>>(
        input_ptr, output_ptr, _nchans, nsamps, _nsamps_per_packet, nsamps_per_load, _scale, _offset);
    CUDA_ERROR_CHECK(cudaStreamSynchronize(_stream));
    BOOST_LOG_TRIVIAL(debug) << "Scaled transpose complete";
}

} //namespace edd
} //namespace effelsberg
} //namespace psrdada_cpp
