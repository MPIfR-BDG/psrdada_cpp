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

    // What are the dimensions...
    // blockDim.x doesn't matter
    // blockDim.y == nchans / fscrunch
    // blockDim.z unused
    // gridDim.x == up to number of timestamps
    // gridDim.y is unused
    // gridDim.z is unused

    static_assert(FBFUSE_NSAMPLES_PER_HEAP % FBFUSE_IB_TSCRUNCH == 0,
        "tscrunch must divide 256");
    static_assert(FBFUSE_NCHANS % FBFUSE_IB_FSCRUNCH == 0,
        "Fscrunch must divide nchannels");

    const int output_size = FBFUSE_NSAMPLES_PER_HEAP/FBFUSE_IB_TSCRUNCH * FBFUSE_NCHANS/FBFUSE_IB_FSCRUNCH;
    volatile __shared__ float accumulation_buffer[FBFUSE_NCHANS/FBFUSE_IB_FSCRUNCH][FBFUSE_NSAMPLES_PER_HEAP];
    volatile __shared__ int8_t output_staging[FBFUSE_NSAMPLES_PER_HEAP/FBFUSE_IB_TSCRUNCH][FBFUSE_NCHANS/FBFUSE_IB_FSCRUNCH];

    //TAFTP
    const int tp = FBFUSE_NSAMPLES_PER_HEAP;
    const int ftp = FBFUSE_NCHANS * tp;
    const int aftp = FBFUSE_IB_NANTENNAS * ftp;
    const int channel_offset = blockIdx.y * FBFUSE_NCHANS;

    for (int timestamp_idx = blockIdx.x; timestamp_idx < ntimestamps; timestamp_idx += gridDim.x)
    {
        for (int sample_idx = threadIdx.x; sample_idx < FBFUSE_NSAMPLES_PER_HEAP; sample_idx += blockDim.x)
        {
            float xx = 0.0f, yy = 0.0f, zz = 0.0f, ww = 0.0f;

            // Must start with the right number of threads in the y dimension
            // blockDim.y = nchans / fscrunch
            for (int channel_idx = FBFUSE_IB_FSCRUNCH * threadIdx.y + channel_offset;
                channel_idx < min(channel_idx + FBFUSE_IB_FSCRUNCH + channel_offset, FBFUSE_NCHANS);
                ++channel_idx)
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
        if (threadIdx.x < FBFUSE_NSAMPLES_PER_HEAP/FBFUSE_IB_TSCRUNCH)
        {
            float val = 0.0f;
            for (int sample_idx = threadIdx.x * FBFUSE_IB_TSCRUNCH; sample_idx < (threadIdx.x + 1) * FBFUSE_IB_TSCRUNCH; ++sample_idx)
            {
                val += accumulation_buffer[threadIdx.y][sample_idx];
            }
            output_staging[threadIdx.x][threadIdx.y] = (int8_t)((val - output_offset)/output_scale);
        }
        __threadfence_block();
        for (int idx = threadIdx.x; idx < output_size; idx += gridDim.x)
        {
            tf_powers[idx * gridDim.y + threadIdx.y] = output_staging[idx][threadIdx.y];
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
    int nthreads_y = _config.nchans() / _config.ib_fscrunch();
    int nthreads_x = 1024 / nthreads_y;
    dim3 block(nthreads_x, nthreads_y);
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

