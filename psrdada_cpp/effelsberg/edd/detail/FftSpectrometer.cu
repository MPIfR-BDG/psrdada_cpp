#include "psrdada_cpp/effelsberg/edd/FftSpectrometer.cuh"
#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/cuda_utils.hpp"
#include "psrdada_cpp/raw_bytes.hpp"
#include <cuda.h>

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {

template <class HandlerType>
FftSpectrometer<HandlerType>::FftSpectrometer(
    std::size_t buffer_bytes,
    std::size_t fft_length,
    std::size_t naccumulate,
    std::size_t nbits,
    float input_level,
    float output_level,
    HandlerType& handler)
    : _buffer_bytes(buffer_bytes)
    , _fft_length(fft_length)
    , _naccumulate(naccumulate)
    , _nbits(nbits)
    , _handler(handler)
    , _fft_plan(0)
    , _call_count(0)
{

    assert(((_nbits == 12) || (_nbits == 8)));
    BOOST_LOG_TRIVIAL(debug)
    << "Creating new FftSpectrometer instance with parameters: \n"
    << "fft_length = " << _fft_length << "\n"
    << "naccumulate = " << _naccumulate;
    std::size_t nsamps_per_buffer = buffer_bytes * 8 / nbits;
    assert(nsamps_per_buffer % _fft_length == 0 /*Number of samples is not multiple of FFT size*/);
    std::size_t n64bit_words = buffer_bytes / sizeof(uint64_t);
    _nchans = _fft_length / 2 + 1;
    int batch = nsamps_per_buffer/_fft_length;
    BOOST_LOG_TRIVIAL(debug) << "Calculating scales and offsets";
    float dof = 2 * _naccumulate;
    float scale = std::pow(input_level * std::sqrt(static_cast<float>(_nchans)), 2);
    float offset = scale * dof;
    float scaling = scale * std::sqrt(2 * dof) / output_level;
    BOOST_LOG_TRIVIAL(debug) << "Correction factors for 8-bit conversion: offset = " << offset << ", scaling = " << scaling;
    BOOST_LOG_TRIVIAL(debug) << "Generating FFT plan";
    int n[] = {static_cast<int>(_fft_length)};
    CUFFT_ERROR_CHECK(cufftPlanMany(&_fft_plan, 1, n, NULL, 1, _fft_length,
        NULL, 1, _fft_length/2 + 1, CUFFT_R2C, batch));
    cufftSetStream(_fft_plan, _proc_stream);
    BOOST_LOG_TRIVIAL(debug) << "Allocating memory";
    _raw_voltage_db.resize(n64bit_words);
    BOOST_LOG_TRIVIAL(debug) << "Input voltages size (in 64-bit words): " << _raw_voltage_db.size();
    _unpacked_voltage.resize(nsamps_per_buffer);
    BOOST_LOG_TRIVIAL(debug) << "Unpacked voltages size (in samples): " << _unpacked_voltage.size();
    _channelised_voltage.resize(_nchans * batch);
    BOOST_LOG_TRIVIAL(debug) << "Channelised voltages size: " << _channelised_voltage.size();
    _power_db.resize(_nchans * batch / _naccumulate);
    BOOST_LOG_TRIVIAL(debug) << "Powers size: " << _power_db.size();
    _host_power_db.resize(_power_db.size());
    CUDA_ERROR_CHECK(cudaStreamCreate(&_h2d_stream));
    CUDA_ERROR_CHECK(cudaStreamCreate(&_proc_stream));
    CUDA_ERROR_CHECK(cudaStreamCreate(&_d2h_stream));
    CUFFT_ERROR_CHECK(cufftSetStream(_fft_plan, _proc_stream));
    _unpacker.reset(new Unpacker(_proc_stream));
    _detector.reset(new DetectorAccumulator(_nchans, _naccumulate,
        scaling, offset, _proc_stream));
}

template <class HandlerType>
FftSpectrometer<HandlerType>::~FftSpectrometer()
{
    BOOST_LOG_TRIVIAL(debug) << "Destroying FftSpectrometer";
    if (!_fft_plan)
        cufftDestroy(_fft_plan);
    cudaStreamDestroy(_h2d_stream);
    cudaStreamDestroy(_proc_stream);
    cudaStreamDestroy(_d2h_stream);
}

template <class HandlerType>
void FftSpectrometer<HandlerType>::init(RawBytes& block)
{
    BOOST_LOG_TRIVIAL(debug) << "FftSpectrometer init called";
    _handler.init(block);
}

template <class HandlerType>
void FftSpectrometer<HandlerType>::process(
    thrust::device_vector<RawVoltageType> const& digitiser_raw,
    thrust::device_vector<IntegratedPowerType>& detected)
{
    BOOST_LOG_TRIVIAL(debug) << "Unpacking raw voltages";
    switch (_nbits)
    {
        case 8:  _unpacker->unpack<8>(digitiser_raw, _unpacked_voltage); break;
        case 12: _unpacker->unpack<12>(digitiser_raw, _unpacked_voltage); break;
        default: throw std::runtime_error("Unsupported number of bits");
    }
    BOOST_LOG_TRIVIAL(debug) << "Performing FFT";
    UnpackedVoltageType* _unpacked_voltage_ptr = thrust::raw_pointer_cast(_unpacked_voltage.data());
    ChannelisedVoltageType* _channelised_voltage_ptr = thrust::raw_pointer_cast(_channelised_voltage.data());
    CUFFT_ERROR_CHECK(cufftExecR2C(_fft_plan,
        (cufftReal*) _unpacked_voltage_ptr,
        (cufftComplex*) _channelised_voltage_ptr));
    CUDA_ERROR_CHECK(cudaStreamSynchronize(_proc_stream));
    _detector->detect(_channelised_voltage, detected);
}

template <class HandlerType>
bool FftSpectrometer<HandlerType>::operator()(RawBytes& block)
{
    ++_call_count;
    BOOST_LOG_TRIVIAL(debug) << "FftSpectrometer operator() called (count = " << _call_count << ")";
    assert(block.used_bytes() == _buffer_bytes /* Unexpected buffer size */);

    CUDA_ERROR_CHECK(cudaStreamSynchronize(_h2d_stream));
    _raw_voltage_db.swap();

    CUDA_ERROR_CHECK(cudaMemcpyAsync(static_cast<void*>(_raw_voltage_db.a_ptr()),
        static_cast<void*>(block.ptr()), block.used_bytes(),
        cudaMemcpyHostToDevice, _h2d_stream));

    if (_call_count == 1)
    {
        return false;
    }

    // Synchronize all streams
    CUDA_ERROR_CHECK(cudaStreamSynchronize(_proc_stream));
    _power_db.swap();
    process(_raw_voltage_db.b(), _power_db.a());

    if (_call_count == 2)
    {
        return false;
    }

    CUDA_ERROR_CHECK(cudaStreamSynchronize(_d2h_stream));
    _host_power_db.swap();
    CUDA_ERROR_CHECK(cudaMemcpyAsync(
        static_cast<void*>(_host_power_db.a_ptr()),
        static_cast<void*>(_power_db.b_ptr()),
        _power_db.size() * sizeof(IntegratedPowerType),
        cudaMemcpyDeviceToHost,
        _d2h_stream));
    
    if (_call_count == 3)
    {
        return false;
    }   

    //Wrap _detected_host_previous in a RawBytes object here;
    RawBytes bytes(reinterpret_cast<char*>(_host_power_db.b_ptr()),
        _host_power_db.size() * sizeof(IntegratedPowerType),
        _host_power_db.size() * sizeof(IntegratedPowerType));
    BOOST_LOG_TRIVIAL(debug) << "Calling handler";
    // The handler can't do anything asynchronously without a copy here 
    // as it would be unsafe (given that it does not own the memory it 
    // is being passed).
    return _handler(bytes);
}

} //edd
} //effelsberg
} //psrdada_cpp


