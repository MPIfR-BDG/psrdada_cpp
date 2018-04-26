#include "psrdada_cpp/effelsberg/edd/eddfft.cuh"
#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/cuda_utils.hpp"
#include "psrdada_cpp/raw_bytes.hpp"

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {

template <class HandlerType>
SimpleFFTSpectrometer<HandlerType>::SimpleFFTSpectrometer(
    int fft_length,
    int naccumulate,
    int nbits,
    HandlerType& handler)
    : _fft_length(fft_length)
    , _naccumulate(naccumulate)
    , _nbits(nbits)
    , _handler(handler)
    , _first_block(true)
    , _nsamps(0)
    , _fft_plan(0)
{
    //cudaStreamCreate(&_h2d_stream);
    //cudaStreamCreate(&_proc_stream);
    //cudaStreamCreate(&_d2h_stream);
}

template <class HandlerType>
SimpleFFTSpectrometer<HandlerType>::~SimpleFFTSpectrometer()
{
    if (!_fft_plan)
        cufftDestroy(_fft_plan);
    //cudaStreamDestroy(_h2d_stream);
    //cudaStreamDestroy(_proc_stream);
    //cudaStreamDestroy(_d2h_stream);
}

template <class HandlerType>
void SimpleFFTSpectrometer<HandlerType>::init(RawBytes& block)
{
    _handler.init(block);
}

template <class HandlerType>
void SimpleFFTSpectrometer<HandlerType>::launch_processing_kernels(thrust::device_vector<uint64_t> const& input)
{

}

template <class HandlerType>
bool SimpleFFTSpectrometer<HandlerType>::operator()(RawBytes& block)
{
    int nsamps_in_block = 8 * block.used_bytes() / _nbits;
    int nchans = _fft_length / 2 + 1;
    if (_first_block)
    {
        _nsamps = nsamps_in_block;
        int n64bit_words = 3 * _nsamps / 16;
        if (_nsamps % _fft_length != 0)
        {
            throw std::runtime_error("Number of samples is not multiple of FFT size");
        }
        int batch = _nsamps/_fft_length;

        // Only do these things once
        int n[] = {_fft_length};
        CUFFT_ERROR_CHECK(cufftPlanMany(&_fft_plan, 1, n, NULL, 1, _fft_length,
            NULL, 1, _fft_length/2 + 1, CUFFT_R2C, batch));
        cufftSetStream(_fft_plan, _proc_stream);
        _edd_raw.resize(n64bit_words);
        _edd_unpacked.resize(_nsamps);
        _channelised.resize(nchans * batch);
        _detected.resize(nchans * batch / _naccumulate);
        _detected_host.resize(nchans * batch / _naccumulate);
        _first_block = false;

        //The first memcopy must be blocking
        //cudaMemcpy((char*) _edd_raw_ptr, block.ptr(), block.used_bytes(), cudaMemcpyHostToDevice);
        //CUDA_ERROR_CHECK(cudaDeviceSynchronize());
        //launch_processing_kernels();
        return false;
    }


    //cudaStreamSynchronize(_h2d_stream);
    //cudaStreamSynchronize(_proc_stream);
    //cudaStreamSynchronize(_d2h_stream);
    //cudaMemcpyAsync((char*) _edd_raw_ptr, block.ptr(), block.used_bytes(), cudaMemcpyHostToDevice, _d2h_stream);


    // --- pass 2 ---
    // Start async memcpy into next buffer in stream 0
    // Process previous block in stream 1
    // return



    // --- pass 3 ---
    // sync streams 1 & 0
    // start async memcpy into output block in stream 2
    // start async memcpy into next buffer in stream 0
    // Process previous block in stream 1

    // --- pass 4 ---
    // sync streams 2, 1 & 0
    // start async memcpy into output block in stream 2
    // start async memcpy into next buffer in stream 0
    // Process previous block in stream 1


    if (_nsamps != nsamps_in_block)
    {
        throw std::runtime_error("Received incomplete block");
    }

    uint64_t* _edd_raw_ptr = thrust::raw_pointer_cast(_edd_raw.data());
    float* _edd_unpacked_ptr = thrust::raw_pointer_cast(_edd_unpacked.data());
    cudaMemcpy((char*) _edd_raw_ptr, block.ptr(), block.used_bytes(), cudaMemcpyHostToDevice);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    if (_nbits == 12)
    {
        int nblocks = _edd_raw.size() / NTHREADS_UNPACK;
        kernels::unpack_edd_12bit_to_float32<<< nblocks, NTHREADS_UNPACK>>>(_edd_raw_ptr, _edd_unpacked_ptr, _edd_raw.size());
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    }
    else if (_nbits == 8)
    {
        throw std::runtime_error("Only 12-bit mode supported");
    }
    else
    {
        throw std::runtime_error("Only 12-bit mode supported");
    }

    cufftComplex* _channelised_ptr = thrust::raw_pointer_cast(_channelised.data());
    //cufftSetStream(_fft_plan, stream);
    CUFFT_ERROR_CHECK(cufftExecR2C(_fft_plan, (cufftReal*)_edd_unpacked_ptr, _channelised_ptr));

    float* _detected_ptr = thrust::raw_pointer_cast(_detected.data());
    kernels::detect_and_accumulate<<<1024, 1024>>>(_channelised_ptr, _detected_ptr, nchans, _nsamps/_fft_length, 64);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    cudaMemcpy((char*) _edd_raw_ptr, block.ptr(), block.used_bytes(), cudaMemcpyHostToDevice);

    RawBytes bytes((char*) thrust::raw_pointer_cast(_detected_host),
        _detected_host.size()*sizeof(float),
        _detected_host.size()*sizeof(float));
    return _handler(bytes);
}

} //edd
} //effelsberg
} //psrdada_cpp


