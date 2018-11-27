#include "psrdada_cpp/meerkat/fbfuse/IncoherentBeamformer.cuh"
#include "psrdada_cpp/cuda_utils.hpp"
#include <cassert>

#define FBFUSE_IB_MAX_NCHANS_OUT_PER_BLOCK 16

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

    // What are the dimensions...
    // blockDim.x doesn't matter
    // blockDim.y == nchans / fscrunch
    // blockDim.z unused
    // gridDim.x == up to number of timestamps
    // gridDim.y // Could use this for channel groups to keep shared memory size down
    // gridDim.z is unused

    static_assert(FBFUSE_NSAMPLES_PER_HEAP % FBFUSE_IB_TSCRUNCH == 0,
        "tscrunch must divide 256");
    static_assert(FBFUSE_IB_MAX_NCHANS_OUT_PER_BLOCK % FBFUSE_IB_FSCRUNCH == 0,
        "Fscrunch must divide nchannels");

    const int nchans_output_total = FBFUSE_NCHANS / FBFUSE_IB_FSCRUNCH;
    const int nchans_output_block = FBFUSE_IB_MAX_NCHANS_OUT_PER_BLOCK/FBFUSE_IB_FSCRUNCH;
    const int nsamps_output = FBFUSE_NSAMPLES_PER_HEAP/FBFUSE_IB_TSCRUNCH;
    volatile __shared__ float accumulation_buffer[nchans_output_block][FBFUSE_NSAMPLES_PER_HEAP];
    volatile __shared__ int8_t output_staging[nsamps_output][nchans_output_block];

    //TAFTP
    const int tp = FBFUSE_NSAMPLES_PER_HEAP;
    const int ftp = FBFUSE_NCHANS * tp;
    const int aftp = FBFUSE_IB_NANTENNAS * ftp;
    const int channel_offset = blockIdx.y * FBFUSE_IB_MAX_NCHANS_OUT_PER_BLOCK;

    for (int timestamp_idx = blockIdx.x; timestamp_idx < ntimestamps; timestamp_idx += gridDim.x)
    {
        for (int sample_idx = threadIdx.x; sample_idx < FBFUSE_NSAMPLES_PER_HEAP; sample_idx += blockDim.x)
        {
            float xx = 0.0f, yy = 0.0f, zz = 0.0f, ww = 0.0f;

            // Must start with the right number of threads in the y dimension
            // blockDim.y = nchans / fscrunch
	    const int start_chan = FBFUSE_IB_FSCRUNCH * threadIdx.y + channel_offset;
            for (int channel_idx = start_chan; channel_idx < start_chan + FBFUSE_IB_FSCRUNCH; ++channel_idx)
            {
                for (int antenna_idx = 0; antenna_idx < FBFUSE_IB_NANTENNAS; ++antenna_idx)
                {
                    int input_index = timestamp_idx * aftp + antenna_idx * ftp + channel_idx * tp + sample_idx;
                    char4 ant = taftp_voltages[input_index];
                    xx += ((float) ant.x) * ant.x;
                    yy += ((float) ant.y) * ant.y;
                    zz += ((float) ant.z) * ant.z;
                    ww += ((float) ant.w) * ant.w;
                }
            }
            accumulation_buffer[threadIdx.y][sample_idx] = (xx + yy + zz + ww);
        }
        __threadfence_block();
        for (int output_sample_idx = threadIdx.x; output_sample_idx < nsamps_output; output_sample_idx += blockDim.x)
        {
            float val = 0.0f;
            for (int sample_idx = output_sample_idx * FBFUSE_IB_TSCRUNCH; sample_idx < (output_sample_idx + 1) * FBFUSE_IB_TSCRUNCH; ++sample_idx)
            {
                val += accumulation_buffer[threadIdx.y][sample_idx];
            }
            output_staging[output_sample_idx][threadIdx.y] = (int8_t)((val - output_offset)/output_scale);
        }
        __syncthreads();
	const int output_offset = timestamp_idx * nsamps_output * nchans_output_total;
        for (int idx = threadIdx.y; idx < nsamps_output; idx += blockDim.y)
        {
            for (int output_chan_idx = threadIdx.x; output_chan_idx < nchans_output; output_chan_idx += blockDim.x)
            {
                tf_powers[output_offset + idx * nchans_output_total + output_chan_idx + channel_offset] = output_staging[idx][output_chan_idx];
	    }
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
    BOOST_LOG_TRIVIAL(debug) << "Executing coherent beamforming";
    assert(input.size() % _size_per_aftp_block == 0 /* Input is not a multiple of AFTP blocks*/);
    std::size_t ntimestamps = input.size() / _size_per_aftp_block;
    std::size_t output_size = (input.size() / _config.ib_nantennas()
	/ _config.npol() / _config.ib_tscrunch() / _config.ib_fscrunch());
    BOOST_LOG_TRIVIAL(debug) << "Resizing output buffer from "
    << output.size() << " to " << output_size
    << " elements";
    output.resize(output_size);
    assert(FBFUSE_IB_MAX_NCHANS_OUT_PER_BLOCK % _config.ib_fscrunch() == 0 /* IB fscrunch must divide the number of output channels per block*/);

    // The incoherent beamforming kernel can only handle 32 output channels per
    // block. As such we use the gridDim.y to handle blocks of 32 channels.
    int nchans_out_total = _config.nchans() / _config.ib_fscrunch();
    int nchans_groups = 1;
    if (nchans_out_total > FBFUSE_IB_MAX_NCHANS_OUT_PER_BLOCK)
    {
        // Assumes that nchans is always a power of two.
        nchans_groups = nchans_out_total / FBFUSE_IB_MAX_NCHANS_OUT_PER_BLOCK;
    }
    BOOST_LOG_TRIVIAL(debug) << "IB kernel using " << nchans_groups << " channel groups";
    int nthreads_y = nchans_out_total / nchans_groups;
    int nthreads_x = 1024 / nthreads_y;
    dim3 block(nthreads_x, nthreads_y);
    dim3 grid(ntimestamps, nchans_groups);
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

