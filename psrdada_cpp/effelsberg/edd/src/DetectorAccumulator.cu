#include "psrdada_cpp/effelsberg/edd/DetectorAccumulator.cuh"
#include "psrdada_cpp/cuda_utils.hpp"

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {
namespace kernels {

__global__
void detect_and_accumulate(float2 const* __restrict__ in, int8_t* __restrict__ out,
    int nchans, int nsamps, int naccumulate, float scale, float offset)
{
    for (int block_idx = blockIdx.x; block_idx < nsamps/naccumulate; block_idx += gridDim.x)
    {
        int read_offset = block_idx * naccumulate * nchans;
        int write_offset = block_idx * nchans;
        for (int chan_idx = threadIdx.x; chan_idx < nchans; chan_idx += blockDim.x)
        {
            float sum = 0.0f;
            for (int ii=0; ii < naccumulate; ++ii)
            {
                float2 tmp = in[read_offset + chan_idx + ii*nchans];
                float x = tmp.x * tmp.x;
                float y = tmp.y * tmp.y;
                sum += x + y;
            }
            out[write_offset + chan_idx] = (int8_t) (sum - offset)/scale;
        }
    }
}

} //namespace kernels

DetectorAccumulator::DetectorAccumulator(
    int nchans, int tscrunch, float scale,
    float offset, cudaStream_t stream)
    : _nchans(nchans)
    , _tscrunch(tscrunch)
    , _scale(scale)
    , _offset(offset)
    , _stream(stream)
{

}

DetectorAccumulator::~DetectorAccumulator()
{

}

DetectorAccumulator::detect(InputType const& input, OutputType& output)
{
    assert(input.size() % (_nchans * _tscrunch) == 0 /* Input is not a multiple of _nchans * _tscrunch*/);
    output.resize(input.size()/_tscrunch);
    int nsamps = input.size() / _nchans;
    float const* input_ptr = thrust::raw_pointer_cast(input.data());
    float* output_ptr = thrust::raw_pointer_cast(output.data());
    kernels::detect_and_accumulate<<<1024, 1024, 0, _stream>>>(
        input_ptr, output_ptr, _nchans, nsamps, _tscrunch, _scale, _offset);
    CUDA_ERROR_CHECK(cudaStreamSynchronize(_stream));
}

} //namespace edd
} //namespace effelsberg
} //namespace psrdada_cpp
