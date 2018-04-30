#include "psrdada_cpp/effelsberg/edd/eddfft.cuh"
#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/cuda_utils.hpp"
#include "psrdada_cpp/raw_bytes.hpp"

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {

template <class HandlerType>
SimpleFFTSpectrometer<HandlerType>::SimpleFFTSpectrometer(
    int nsamps_per_block,
    int fft_length,
    int naccumulate,
    int nbits,
    HandlerType& handler)
    : _nsamps(nsamps_per_block)
    , _fft_length(fft_length)
    , _naccumulate(naccumulate)
    , _nbits(nbits)
    , _handler(handler)
    , _fft_plan(0)
{
    BOOST_LOG_TRIVIAL(debug) << "Creating new SimpleFFTSpectrometer instance with parameters: \n"
    << "fft_length = " << _fft_length << "\n"
    << "naccumulate = " << _naccumulate;
    //cudaStreamCreate(&_h2d_stream);
    //cudaStreamCreate(&_proc_stream);
    //cudaStreamCreate(&_d2h_stream);

    int n64bit_words = 3 * _nsamps / 16;
    if (_nsamps % _fft_length != 0)
    {
        throw std::runtime_error("Number of samples is not multiple of FFT size");
    }
    int batch = _nsamps/_fft_length;

    BOOST_LOG_TRIVIAL(debug) << "Generating FFT plan";
        // Only do these things once
    int n[] = {_fft_length};
    CUFFT_ERROR_CHECK(cufftPlanMany(&_fft_plan, 1, n, NULL, 1, _fft_length,
        NULL, 1, _fft_length/2 + 1, CUFFT_R2C, batch));

    BOOST_LOG_TRIVIAL(debug) << "Allocating memory";
        //cufftSetStream(_fft_plan, _proc_stream);
    _edd_raw.resize(n64bit_words);
    _edd_unpacked.resize(_nsamps);
    _channelised.resize(nchans * batch);
    _detected.resize(nchans * batch / _naccumulate);
    _detected_host.resize(nchans * batch / _naccumulate);

    //The first memcopy must be blocking
    //cudaMemcpy((char*) _edd_raw_ptr, block.ptr(), block.used_bytes(), cudaMemcpyHostToDevice);
    //CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    //launch_processing_kernels();

}

template <class HandlerType>
SimpleFFTSpectrometer<HandlerType>::~SimpleFFTSpectrometer()
{
    BOOST_LOG_TRIVIAL(debug) << "Destroying SimpleFFTSpectrometer";
    if (!_fft_plan)
        cufftDestroy(_fft_plan);
    //cudaStreamDestroy(_h2d_stream);
    //cudaStreamDestroy(_proc_stream);
    //cudaStreamDestroy(_d2h_stream);
}

template <class HandlerType>
void SimpleFFTSpectrometer<HandlerType>::init(RawBytes& block)
{
    BOOST_LOG_TRIVIAL(debug) << "SimpleFFTSpectrometer init called";
    _handler.init(block);
}


template <class HandlerType>
bool SimpleFFTSpectrometer<HandlerType>::operator()(RawBytes& block)
{
    BOOST_LOG_TRIVIAL(debug) << "SimpleFFTSpectrometer operator() called";
    int nsamps_in_block = 8 * block.used_bytes() / _nbits;
    int nchans = _fft_length / 2 + 1;
    BOOST_LOG_TRIVIAL(debug) << nsamps_in_block << " samples in RawBytes block";

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

    BOOST_LOG_TRIVIAL(debug) << "Copying RawBytes contents to GPU";
    uint64_t* _edd_raw_ptr = thrust::raw_pointer_cast(_edd_raw.data());
    float* _edd_unpacked_ptr = thrust::raw_pointer_cast(_edd_unpacked.data());
    cudaMemcpy((char*) _edd_raw_ptr, block.ptr(), block.used_bytes(), cudaMemcpyHostToDevice);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    BOOST_LOG_TRIVIAL(debug) << "Unpacking digitiser data";
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

    BOOST_LOG_TRIVIAL(debug) << "Performing FFT";
    cufftComplex* _channelised_ptr = thrust::raw_pointer_cast(_channelised.data());
    //cufftSetStream(_fft_plan, stream);
    CUFFT_ERROR_CHECK(cufftExecR2C(_fft_plan, (cufftReal*)_edd_unpacked_ptr, _channelised_ptr));

    BOOST_LOG_TRIVIAL(debug) << "Detecting and accumulating";
    float* _detected_ptr = thrust::raw_pointer_cast(_detected.data());
    kernels::detect_and_accumulate<<<1024, 1024>>>(_channelised_ptr, _detected_ptr, nchans, _nsamps/_fft_length, 64);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    BOOST_LOG_TRIVIAL(debug) << "Copying resultant data to host";
    cudaMemcpy((char*) _edd_raw_ptr, block.ptr(), block.used_bytes(), cudaMemcpyHostToDevice);

    RawBytes bytes((char*) thrust::raw_pointer_cast(_detected_host.data()),
        _detected_host.size()*sizeof(float),
        _detected_host.size()*sizeof(float));
    BOOST_LOG_TRIVIAL(debug) << "Calling handler";
    return _handler(bytes);
}

} //edd
} //effelsberg
} //psrdada_cpp


