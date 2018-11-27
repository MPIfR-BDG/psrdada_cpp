#include "psrdada_cpp/meerkat/fbfuse/IncoherentBeamformer.cuh"
#include "psrdada_cpp/cuda_utils.hpp"
#include <cassert>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace kernels {

__global__
void icbf_taftp_general_k(
    char4 const* __restrict__ taftp_voltages,
    int8_t* __restrict__ tf_powers,
    float output_scale,
    float output_offset,
    int ntimestamps)
{
    //TAFTP
    const int tp = FBFUSE_NSAMPLES_PER_HEAP;
    const int ftp = FBFUSE_NCHANS * tp;
    const int aftp = FBFUSE_IB_NANTENNAS * ftp;
    const int nchans_out = FBFUSE_NCHANS / FBFUSE_IB_FSCRUNCH;
    const int nsamps_out = FBFUSE_NSAMPLES_PER_HEAP / FBFUSE_IB_TSCRUNCH;
    volatile __shared__ float acc_buffer[FBFUSE_NSAMPLES_PER_HEAP];
    volatile __shared__ int8_t output_buffer[nsamps_out * nchans_out];

    for (int timestamp_idx = blockIdx.x; timestamp_idx < ntimestamps; timestamp_idx += gridDim.x)
    {
        for (int start_channel_idx = 0; start_channel_idx < FBFUSE_NCHANS; start_channel_idx += FBFUSE_IB_FSCRUNCH)
        {
	    float xx = 0.0f, yy = 0.0f, zz = 0.0f, ww = 0.0f;
            for (int sub_channel_idx = start_channel_idx; sub_channel_idx < (start_channel_idx + 1) * FBFUSE_IB_FSCRUNCH; ++sub_channel_idx)
            {
                for (int antenna_idx = 0; antenna_idx < FBFUSE_IB_NANTENNAS; ++antenna_idx)
                {
                    int input_index = timestamp_idx * aftp + antenna_idx * ftp + sub_channel_idx * tp + threadIdx.x;
                    char4 ant = taftp_voltages[input_index];
                    xx += ((float) ant.x) * ant.x;
                    yy += ((float) ant.y) * ant.y;
                    zz += ((float) ant.z) * ant.z;
                    ww += ((float) ant.w) * ant.w;
                }
            }
            float val = (xx + yy + zz + ww);
            acc_buffer[threadIdx.x] = val;
            __syncthreads();
            for (int ii = 1; ii < FBFUSE_IB_TSCRUNCH; ++ii)
            {
                int idx = threadIdx.x + ii;
                if (idx < FBFUSE_NSAMPLES_PER_HEAP)
                {
                    val += acc_buffer[idx];
                }
            }
            if (threadIdx.x % FBFUSE_IB_TSCRUNCH == 0)
            {
                int output_buffer_idx = threadIdx.x/FBFUSE_IB_TSCRUNCH * nchans_out + start_channel_idx/FBFUSE_IB_FSCRUNCH;
                output_buffer[output_buffer_idx] = (int8_t)((val - output_offset)/output_scale);
            }
            __syncthreads();
        }
        int output_offset = timestamp_idx * nsamps_out * nchans_out;
        for (int idx = threadIdx.x; idx < nsamps_out * FBFUSE_NCHANS/FBFUSE_IB_FSCRUNCH; idx += blockDim.x)
        {
            tf_powers[output_offset + idx] = output_buffer[idx];
        }
    }
}

} //namespace kernels


IncoherentBeamformer::IncoherentBeamformer(PipelineConfig const& config)
    : _config(config)
    , _size_per_aftp_block(0)
{
    BOOST_LOG_TRIVIAL(debug) << "Constructing IncoherentBeamformer instance";
    _size_per_aftp_block = (_config.npol() * _config.ib_nantennas()
        * _config.nchans() * _config.nsamples_per_heap());
    BOOST_LOG_TRIVIAL(debug) << "Size per AFTP block: " << _size_per_aftp_block;
}

IncoherentBeamformer::~IncoherentBeamformer()
{

}

void IncoherentBeamformer::beamform(VoltageVectorType const& input,
    PowerVectorType& output,
    cudaStream_t stream)
{
    // First work out nsamples and resize output if not done already
    BOOST_LOG_TRIVIAL(debug) << "Executing incoherent beamforming";
    assert(input.size() % _size_per_aftp_block == 0 /* Input is not a multiple of AFTP blocks*/);
    std::size_t ntimestamps = input.size() / _size_per_aftp_block;
    std::size_t output_size = (input.size() / _config.ib_nantennas()
    / _config.npol() / _config.ib_tscrunch() / _config.ib_fscrunch());
    BOOST_LOG_TRIVIAL(debug) << "Resizing output buffer from "
    << output.size() << " to " << output_size
    << " elements";
    output.resize(output_size);
    assert(FBFUSE_IB_MAX_NCHANS_OUT_PER_BLOCK % _config.ib_fscrunch() == 0 /* IB fscrunch must divide the number of output channels per block*/);
    int nthreads_x = FBFUSE_NSAMPLES_PER_HEAP;
    dim3 block(nthreads_x);
    dim3 grid(ntimestamps);
    char2 const* taftp_voltages_ptr = thrust::raw_pointer_cast(input.data());
    int8_t* tf_powers_ptr = thrust::raw_pointer_cast(output.data());
    BOOST_LOG_TRIVIAL(debug) << "Executing incoherent beamforming kernel";
    kernels::icbf_taftp_general_k<<<grid, block, 0, stream>>>(
        (char4 const*) taftp_voltages_ptr,
        tf_powers_ptr,
        _config.ib_power_scaling(),
        _config.ib_power_offset(),
        static_cast<int>(ntimestamps));
    CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
    BOOST_LOG_TRIVIAL(debug) << "Incoherent beamforming kernel complete";
}

} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

