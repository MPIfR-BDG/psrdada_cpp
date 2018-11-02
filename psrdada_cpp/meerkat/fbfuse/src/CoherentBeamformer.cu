#include "psrdada_cpp/meerkat/fbfuse/CoherentBeamformer.cuh"
#include "psrdada_cpp/cuda_utils.hpp"
#include <cassert>

#define FBFUSE_CB_WARP_SIZE 32
#define FBFUSE_CB_NTHREADS 1024
#define FBFUSE_CB_NWARPS_PER_BLOCK (FBFUSE_CB_NTHREADS / FBFUSE_CB_WARP_SIZE)

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace kernels {

__forceinline__ __device__
void dp4a(int &c, const int &a, const int &b) {
#if __CUDA_ARCH__ >= 610
  asm("dp4a.s32.s32 %0, %1, %2, %3;" : "+r"(c) : "r"(a), "r"(b), "r"(c));
#else
  char4 &a4 = *((char4*)&a);
  char4 &b4 = *((char4*)&b);
  c += a4.x*b4.x;
  c += a4.y*b4.y;
  c += a4.z*b4.z;
  c += a4.w*b4.w;
#endif
}

__forceinline__ __device__
int2 int2_transpose(int2 const &input)
{
    char2x4 a;
    char4x2 b;
    a = (*(char2x4*)&input);
    b.x.x = a.x.x;
    b.x.y = a.y.x;
    b.x.z = a.z.x;
    b.x.w = a.w.x;
    b.y.x = a.x.y;
    b.y.y = a.y.y;
    b.y.z = a.z.y;
    b.y.w = a.w.y;
    return (*(int2*)&b);
}

/**
 * @brief      Perform beamforming followed by detection and integration in time.
 *
 * @param      ftpa_voltages  Raw voltages in antenna, polarisation, time, frequency order (fastest to slowest)
 * @param      fbpa_weights   Beamforming weights in antenna, time, beam, frequency order (fastest to slowest)
 * @param      tbtf_powers     Output detected integrated powers in frequency, time, beam order (fastest to slowest)
 */
__global__
void bf_aptf_general_k(
    int2 const* __restrict__ ftpa_voltages,
    int2 const* __restrict__ fbpa_weights,
    char* __restrict__ tbtf_powers,
    float output_scale,
    float output_offset,
    int nsamples)
{
    /**
     * Perform compile time checks on requested beamforming parameters.
     */
    static_assert(FBFUSE_CB_NBEAMS%FBFUSE_CB_WARP_SIZE==0,
        "Kernel can only process a multiple of 32 beams.");
    // This can no longer be a static assert as the NSAMPLES is no longer fixed
    // static_assert(NSAMPLES%FBFUSE_CB_NSAMPLES_PER_BLOCK==0,
    //    "Kernel can only process a multiple of (NWARPS_PER_BLOCK * FBFUSE_CB_TSCRUNCH) samples.");
    static_assert(FBFUSE_CB_NTHREADS%FBFUSE_CB_WARP_SIZE==0,
        "Number of threads must be an integer multiple of FBFUSE_CB_WARP_SIZE.");
    static_assert(FBFUSE_CB_NANTENNAS%4==0,
        "Number of antennas must be a multiple of 4.");
    static_assert(FBFUSE_CB_NANTENNAS<=128,
        "Number of antennas must be less than or equal to 128.");
    /**
     * Allocated shared memory to store beamforming weights and temporary space for antenna data.
     */
    __shared__ int2 shared_apb_weights[FBFUSE_CB_NANTENNAS/4][FBFUSE_CB_WARP_SIZE];
    __shared__ int2 shared_antennas[FBFUSE_CB_NTHREADS/FBFUSE_CB_WARP_SIZE][FBFUSE_CB_NANTENNAS/4];
    int const warp_idx = threadIdx.x / 0x20;
    int const lane_idx = threadIdx.x & 0x1f;

    /**
     * Each warp processes 32 beams (i.e. one beam per lane).
     */
    int const start_beam_idx = blockIdx.z * FBFUSE_CB_WARP_SIZE;

    /**
     * Complex multiply accumulators
     */
    int xx, yy, xy, yx;

    float power = 0.0f;
    int2 antennas, weights;
    int antenna_group_idx;

    /**
     * Here we load all the beamforming weights neccessary for this block. Implicit assumption here is that we do not
     * need to change the weights over the timescale of the data processed in one block. This is almost certainly OK
     * if the input data has already been rotated to telescope boresight and we are only applying parallactic angle
     * tracking updates.
     *
     * The global load is coalesced 8-byte (vectorised int2).
     */
    int const fbpa_weights_offset = FBFUSE_CB_NANTENNAS/4 * (FBFUSE_CB_NBEAMS * blockIdx.y + (start_beam_idx + warp_idx));
    for (antenna_group_idx = lane_idx; antenna_group_idx < FBFUSE_CB_NANTENNAS/4; antenna_group_idx += FBFUSE_CB_WARP_SIZE)
    {
      shared_apb_weights[antenna_group_idx][warp_idx] = int2_transpose(fbpa_weights[fbpa_weights_offset + antenna_group_idx]);
    }

    //wait for all weights to load.
    __syncthreads();

    /**
     * Below is the main loop of the kernel. Here the kernel reads all the antennas for a given sample and
     * computes 32 beams. Each thread computes only 1 beam and access to all the antennas required for that
     * computation is achieved via a shared memory broadcasts.
     */
    int sample_offset = FBFUSE_CB_TSCRUNCH * (blockIdx.x * FBFUSE_CB_NWARPS_PER_BLOCK + warp_idx);
    for (int sample_idx = sample_offset; sample_idx < (sample_offset + FBFUSE_CB_TSCRUNCH); ++sample_idx)
    {
        int ftpa_voltages_partial_idx = FBFUSE_CB_NANTENNAS/4 * FBFUSE_NPOL * (nsamples * blockIdx.y + sample_idx);
        for (int pol_idx=0; pol_idx < FBFUSE_NPOL; ++pol_idx)
        {
            // Set the complex accumulator to zero before adding the next polarisation
            xx = 0;
            yy = 0;
            xy = 0;
            yx = 0;

           /**
            * Load all antennas antennas required for this sample into shared memory.
            * Without an outer loop to allow for more antennas (which would also require more shared memory),
            * this kernel is limited to a max of 32 * 4 = 128 antennas in a sub-array.
            */
            if (lane_idx < FBFUSE_CB_NANTENNAS/4)
            {
                shared_antennas[warp_idx][lane_idx] = int2_transpose(ftpa_voltages[ftpa_voltages_partial_idx + lane_idx + FBFUSE_CB_NANTENNAS/4 * pol_idx]);
            }
            __threadfence_block();
            for (antenna_group_idx=0; antenna_group_idx < FBFUSE_CB_NANTENNAS/4; ++antenna_group_idx)
            {
                //broadcast load 4 antennas
                antennas = shared_antennas[warp_idx][antenna_group_idx];
                //load corresponding 4 weights
                weights = shared_apb_weights[antenna_group_idx][lane_idx];
                //dp4a multiply add
                dp4a(xx,weights.x,antennas.x);
                dp4a(yy,weights.y,antennas.y);
                dp4a(xy,weights.x,antennas.y);
                dp4a(yx,weights.y,antennas.x);
            }
            int r = xx - yy;
            int i = xy + yx;
            //be careful of overflow
            power += (float)(r*r + i*i);
        }
    }

    /**
     * As we have looped over both polarisation and sample in the above loop we are now free to simply
     * write back to global memory. Here we write back uncoalesced to get the data in time beam order.
     * The performance penalty here is very small compared to the compute time in the rest of the kernel
     * as the total volume of data being written out is a factor of FBFUSE_CB_TSCRUNCH * FBFUSE_CB_NANTENNAS / FBFUSE_CB_WARP_SIZE
     * smaller than the input (e.g. for 64 antennas and 16 integrated samples this is a factor of 32).
     */

    /** ORIGINAL
    int output_idx = (NWARPS_PER_BLOCK * gridDim.x) * (FBFUSE_CB_NBEAMS * blockIdx.y
        + (start_beam_idx+lane_idx))
    + sample_offset / FBFUSE_CB_TSCRUNCH;
    tbtf_powers[output_idx] = power;
    */

    // Wanted output in BTF order
    // But now need in TBTF order (TODO!!!!!!!)
    /* Original implementation
    int const output_idx = gridDim.y * (((start_beam_idx+lane_idx) * FBFUSE_CB_NWARPS_PER_BLOCK * gridDim.x)
          + (sample_offset / FBFUSE_CB_TSCRUNCH)) + blockIdx.y;
    tbtf_powers[output_idx] = (int8_t) ((power - output_offset) / output_scale);
    */
    int const output_sample_idx = sample_offset / FBFUSE_CB_TSCRUNCH;
    int const tf_size = FBFUSE_CB_NSAMPLES_PER_HEAP * gridDim.y;
    int const btf_size = gridDim.z * FBFUSE_CB_WARP_SIZE * tf_size;
    int const output_idx = (output_sample_idx / FBFUSE_CB_NSAMPLES_PER_HEAP * btf_size
        + (start_beam_idx + lane_idx) * tf_size
        + (output_sample_idx % FBFUSE_CB_NSAMPLES_PER_HEAP) * gridDim.y
        + blockIdx.y);
    tbtf_powers[output_idx] = (int8_t) ((power - output_offset) / output_scale);
}

} //namespace kernels


CoherentBeamformer::CoherentBeamformer(PipelineConfig const& config)
    : _config(config)
    , _size_per_sample(0)
{
    BOOST_LOG_TRIVIAL(debug) << "Constructing CoherentBeamformer instance";
    _size_per_sample = _config.npol() * _config.cb_nantennas() * _config.nchans();
    _expected_weights_size = _config.cb_nbeams() * _config.cb_nantennas() * _config.nchans();
    BOOST_LOG_TRIVIAL(debug) << "Size per sample: " << _size_per_sample;
    BOOST_LOG_TRIVIAL(debug) << "Expected weights size: " << _expected_weights_size;
}

CoherentBeamformer::~CoherentBeamformer()
{

}

void CoherentBeamformer::beamform(VoltageVectorType const& input,
    WeightsVectorType const& weights,
    PowerVectorType& output,
    cudaStream_t stream)
{
    // First work out nsamples and resize output if not done already
    BOOST_LOG_TRIVIAL(debug) << "Executing coherent beamforming";
    assert(input.size() % _size_per_sample == 0);
    std::size_t nsamples = input.size() / _size_per_sample;
    std::size_t output_size = (input.size() / _config.cb_nantennas()
	/ _config.npol() / _config.cb_tscrunch() / _config.cb_fscrunch()
	* _config.cb_nbeams());
    assert(nsamples % FBFUSE_CB_NSAMPLES_PER_BLOCK == 0);
    std::size_t nsamples_out = nsamples / _config.cb_tscrunch();
    assert(nsamples_out % FBFUSE_CB_NSAMPLES_PER_HEAP == 0);
    BOOST_LOG_TRIVIAL(debug) << "Resizing output buffer from "
    << output.size() << " to " << output_size
    << " elements";
    output.resize(output_size);
    assert(weights.size() == _expected_weights_size);
    dim3 grid(nsamples / (FBFUSE_CB_NWARPS_PER_BLOCK * _config.cb_tscrunch()),
        _config.nchans(), _config.cb_nbeams()/FBFUSE_CB_WARP_SIZE);
    char2 const* ftpa_voltages_ptr = thrust::raw_pointer_cast(input.data());
    char2 const* fbpa_weights_ptr = thrust::raw_pointer_cast(weights.data());
    char* tbtf_powers_ptr = thrust::raw_pointer_cast(output.data());
    BOOST_LOG_TRIVIAL(debug) << "Executing beamforming kernel";
    kernels::bf_aptf_general_k<<<grid, FBFUSE_CB_NTHREADS, 0, stream>>>(
        (int2 const*) ftpa_voltages_ptr,
        (int2 const*) fbpa_weights_ptr,
        tbtf_powers_ptr,
        _config.cb_power_scaling(),
        _config.cb_power_offset(),
        static_cast<int>(nsamples));
    CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
    BOOST_LOG_TRIVIAL(debug) << "Beamforming kernel complete";
}

} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

