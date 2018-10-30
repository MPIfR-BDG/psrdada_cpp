#include "psrdada_cpp/meerkat/fbfuse/SplitTranspose.cuh"
#include "psrdada_cpp/cuda_utils.hpp"

#define FBFUSE_ST_MAX_ANTENNAS 32

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace kernels {

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
    __shared__ char2 transpose_buffer[FBFUSE_ST_MAX_ANTENNAS][FBFUSE_NSAMPLES_PER_HEAP][FBFUSE_NPOL];

    //TAFTP (input dimensions)
    const int tp = FBFUSE_NSAMPLES_PER_HEAP * FBFUSE_NPOL;
    const int ftp = nchans * tp;
    const int aftp = total_nantennas * ftp;

    //FTPA
    const int pa = FBFUSE_NPOL * used_nantennas;
    const int tpa = ntimestamps * FBFUSE_NSAMPLES_PER_HEAP * pa;
    int nantennas_sets = ceilf(((float) used_nantennas) / FBFUSE_ST_MAX_ANTENNAS);
    for (int timestamp_idx = blockIdx.x; timestamp_idx < ntimestamps; timestamp_idx += gridDim.x)
    {
        for (int chan_idx = blockIdx.y; chan_idx < nchans; chan_idx += gridDim.y)
        {
            for (int antenna_set_idx = 0; antenna_set_idx < nantennas_sets; ++antenna_set_idx)
            {
                int remaining_antennas = min(used_nantennas - antenna_set_idx * FBFUSE_ST_MAX_ANTENNAS, FBFUSE_ST_MAX_ANTENNAS);
                // Load data into shared memory
                for (int antenna_idx = threadIdx.y; antenna_idx < remaining_antennas; antenna_idx += blockDim.y)
                {
                    int input_antenna_idx = antenna_set_idx * FBFUSE_ST_MAX_ANTENNAS + antenna_idx + start_antenna;
                    for (int samppol_idx = threadIdx.x; samppol_idx < (FBFUSE_NSAMPLES_PER_HEAP * FBFUSE_NPOL); samppol_idx += blockDim.x)
                    {
                        int pol_idx = samppol_idx%FBFUSE_NPOL;
                        int samp_idx = samppol_idx/FBFUSE_NPOL;
                        int input_idx = timestamp_idx * aftp + input_antenna_idx * ftp + chan_idx * tp + samp_idx * FBFUSE_NPOL + pol_idx;
                        transpose_buffer[antenna_idx][samp_idx][pol_idx] = input[input_idx];
                    }
                }
                __syncthreads();
                for (int pol_idx = 0; pol_idx < FBFUSE_NPOL; ++pol_idx)
                {
                    for (int samp_idx = threadIdx.y; samp_idx < FBFUSE_NSAMPLES_PER_HEAP; samp_idx += blockDim.y)
                    {
                        int output_sample_idx = samp_idx + timestamp_idx * FBFUSE_NSAMPLES_PER_HEAP;
                        for (int antenna_idx = threadIdx.x; antenna_idx < remaining_antennas; antenna_idx += blockDim.x)
                        {
                            int output_antenna_idx = antenna_set_idx * FBFUSE_ST_MAX_ANTENNAS + antenna_idx;
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

} //namespace kernels


SplitTranspose::SplitTranspose(PipelineConfig const& config)
    : _config(config)
    , _heap_group_size(0)
    , _output_size_per_heap_group(0)
{
    BOOST_LOG_TRIVIAL(debug) << "Constructing SplitTranspose instance";
    _heap_group_size = (_config.npol()
        * _config.nsamples_per_heap()
        * _config.nchans()
        * _config.total_nantennas());
    BOOST_LOG_TRIVIAL(debug) << "Heap group size: " << _heap_group_size;
    _output_size_per_heap_group = (_config.npol()
        * _config.nsamples_per_heap()
        * _config.nchans()
        * _config.cb_nantennas());
    BOOST_LOG_TRIVIAL(debug) << "Output size per heap group: " << _output_size_per_heap_group;
}

SplitTranspose::~SplitTranspose()
{

}

void SplitTranspose::transpose(VoltageType const& taftp_voltages,
        VoltageType& ftpa_voltages, cudaStream_t stream)
{
    BOOST_LOG_TRIVIAL(debug) << "Performing split transpose";
    BOOST_LOG_TRIVIAL(debug) << "Selecting and transposing "
    << _config.cb_nantennas() << " of " << _config.total_nantennas()
    << " antennas starting from antenna " << _config.cb_antenna_offset();
    // Check sizes
    assert(taftp_voltages.size()%_heap_group_size == 0);
    int nheap_groups = taftp_voltages.size() / _heap_group_size;
    BOOST_LOG_TRIVIAL(debug) << "Number of heap groups: " << nheap_groups;
    // Resize output buffer
    BOOST_LOG_TRIVIAL(debug) << "Resizing output buffer from "
    << ftpa_voltages.size() << " to "
    << _output_size_per_heap_group * nheap_groups;
    ftpa_voltages.resize(_output_size_per_heap_group * nheap_groups);
    dim3 grid(nheap_groups, _config.nchans(), 1);
    dim3 block(512, 1, 1);
    char2 const* input_ptr = thrust::raw_pointer_cast(taftp_voltages.data());
    char2* output_ptr = thrust::raw_pointer_cast(ftpa_voltages.data());
    BOOST_LOG_TRIVIAL(debug) << "Launching split transpose kernel";
    kernels::split_transpose_k<<< grid, block, 0, stream>>>(input_ptr, output_ptr,
        _config.total_nantennas(), _config.cb_nantennas(),
        _config.cb_antenna_offset(), _config.nchans(),
        nheap_groups);
    //Not sure if this should be here, will check later
    CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
    BOOST_LOG_TRIVIAL(debug) << "Split transpose complete";
}


} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp