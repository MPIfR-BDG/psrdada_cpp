#include "psrdada_cpp/effelsberg/rfi_chamber/RSSpectrometer.cuh"
#include "psrdada_cpp/cuda_utils.hpp"
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/fill.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/constant_iterator.h>
#include <cassert>
#include <fstream>
#include <iomanip>

#define PASSTHROUGH_MODE_IQ_SCALING 1.0f/(1<<15)
#define PFB_MODE_NSHIFTS 3
#define PFB_MODE_IQ_SCALING 1.0f/(1<<(30 - PFB_MODE_NSHIFTS))
#define FSW_IMPEDANCE 50.0f

namespace psrdada_cpp {
namespace effelsberg {
namespace rfi_chamber {
namespace kernels {

struct short2_be_to_float2_le
    : public thrust::unary_function<short2, float2>
{
    __host__ __device__
    float2 operator()(short2 in)
    {
        char4 swap;
        char4* in_ptr = (char4*)(&in);
        swap.x = in_ptr->y;
        swap.y = in_ptr->x;
        swap.z = in_ptr->w;
        swap.w = in_ptr->z;
        short2* swap_as_short2 = (short2*)(&swap);
        float2 out;
        out.x = (float) swap_as_short2->x;
        out.y = (float) swap_as_short2->y;
        return out;
    }
};

struct detect_scale
    : public thrust::unary_function<float2, float>
{
    detect_scale(float scale_factor=1)
    : _scale_factor(scale_factor){}

    __host__ __device__
    float operator()(float2 voltage)
    {
        float x = voltage.x * _scale_factor;
        float y = voltage.y * _scale_factor;
        float power = x * x + y * y;
        return power;
    }

    const float _scale_factor;
};

struct detect_magnitude
    : public thrust::unary_function<float2, float>
{
    detect_magnitude(float scale_factor=1)
    : _scale_factor(scale_factor){}

    __device__
    float operator()(float2 voltage)
    {
        float x = voltage.x * _scale_factor;
        float y = voltage.y * _scale_factor;
        float power = x * x + y * y;
        return sqrtf(power);
    }

    const float _scale_factor;
};

struct detect_accumulate
    : public thrust::binary_function<float2, float, float>
{
    detect_accumulate(float scale_factor=1)
    : _scale_factor(scale_factor){}

    __host__ __device__
    float operator()(float2 voltage, float power_accumulator)
    {
        float x = voltage.x * _scale_factor;
        float y = voltage.y * _scale_factor;
        float power = x * x + y * y;
        return power_accumulator + power;
    }

    const float _scale_factor;
};

struct convert_to_dBm
    : public thrust::unary_function<float, float>
{
    convert_to_dBm(float scale_factor=1, float offset=0)
    : _scale_factor(scale_factor)
    , _offset(offset){}

    __device__
    float operator()(float power)
    {
        // Typically _scale_factor here is 1000.0 / (50.0 * naccumulate);
        return 10 * __log10f(power * _scale_factor) + _offset;
    }

    const float _scale_factor;
    const float _offset;
};

} // namespace kernels

// dense histogram using binary search
void histogram(const thrust::device_vector<float2>& input,
    thrust::device_vector<int>& d_hist,
    float min_val,
    float max_val,
    std::size_t nbins)
{
    // sort data to bring equal elements together
    thrust::device_vector<float> magnitudes(input.size());
    thrust::transform(input.begin(), input.end(), magnitudes.begin(),
        kernels::detect_magnitude(PASSTHROUGH_MODE_IQ_SCALING));
    thrust::sort(magnitudes.begin(), magnitudes.end());
    thrust::device_vector<float> bins(nbins);
    float step = (max_val - min_val) / nbins;
    thrust::sequence(bins.begin(), bins.end(), min_val, step);
    // resize histogram storage
    d_hist.resize(nbins);
    // find the end of each bin of values
    thrust::upper_bound(magnitudes.begin(), magnitudes.end(),
                        bins.begin(), bins.end(),
                        d_hist.begin());
    // compute the histogram by taking differences of the cumulative histogram
    thrust::adjacent_difference(d_hist.begin(), d_hist.end(),
                                d_hist.begin());
}

RSSpectrometer::RSSpectrometer(
    std::size_t input_nchans, std::size_t fft_length,
    std::size_t naccumulate, std::size_t nskip,
    std::string filename, float reference_dbm)
    : _input_nchans(input_nchans)
    , _fft_length(fft_length)
    , _naccumulate(naccumulate)
    , _nskip(nskip)
    , _filename(filename)
    , _reference_dbm(reference_dbm)
    , _output_nchans(_fft_length * _input_nchans)
    , _bytes_per_input_spectrum(_input_nchans * sizeof(InputType))
    , _naccumulated(0)
{

    BOOST_LOG_TRIVIAL(info) << "Initialising RSSpectrometer";
    BOOST_LOG_TRIVIAL(info) << "Number of input channels: " << _input_nchans;
    BOOST_LOG_TRIVIAL(info) << "FFT length: " << _fft_length;
    BOOST_LOG_TRIVIAL(info) << "Number of spectra to accumulate: " << _naccumulate;
    BOOST_LOG_TRIVIAL(info) << "Number of DADA blocks to skip: " << _nskip;
    BOOST_LOG_TRIVIAL(info) << "Number of output channels: " << _output_nchans;

    std::size_t total_mem, free_mem;
    CUDA_ERROR_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    BOOST_LOG_TRIVIAL(debug) << "Total GPU memory: " << total_mem << " bytes";
    BOOST_LOG_TRIVIAL(debug) << "Free GPU memory: " << free_mem << " bytes";

    // Memory required for accumulation buffer
    std::size_t accumulator_buffer_bytes = _output_nchans * sizeof(OutputType);
    BOOST_LOG_TRIVIAL(debug) << "Memory required for accumulator buffer: " << accumulator_buffer_bytes << " bytes";
    if (accumulator_buffer_bytes > free_mem)
    {
        throw std::runtime_error("The requested FFT length exceeds the free GPU memory");
    }
    std::size_t mem_budget = static_cast<std::size_t>((free_mem - accumulator_buffer_bytes) * 0.8) ; // Make only 80% of memory available
    BOOST_LOG_TRIVIAL(debug) << "Memory budget: " << mem_budget << " bytes";
    // Memory required per input channel
    std::size_t mem_per_input_channel = (_fft_length *  (sizeof(FftType) * 2 + 2 * sizeof(InputType)));
    BOOST_LOG_TRIVIAL(debug) << "Memory required per input channel: " << mem_per_input_channel << " bytes";
    _chans_per_copy = min(_input_nchans, mem_budget / mem_per_input_channel);
    if (mem_per_input_channel > mem_budget)
    {
	 throw std::runtime_error("The requested FFT length exceeds the free GPU memory");
    }
    BOOST_LOG_TRIVIAL(debug) << "Max possible Nchans per copy: " << mem_budget / mem_per_input_channel;
    while (_input_nchans % _chans_per_copy != 0)
    {
        _chans_per_copy -= 1;
    }
    assert(_chans_per_copy > 0); /** Must be able to process at least 1 channel */
    BOOST_LOG_TRIVIAL(debug) << "Nchannels per GPU transfer: " << _chans_per_copy;
    mem_budget -= _chans_per_copy * mem_per_input_channel;
    BOOST_LOG_TRIVIAL(debug) << "Remaining memory budget: " << mem_budget << " bytes";

    // Resize all buffers.
    BOOST_LOG_TRIVIAL(debug) << "Resizing all memory buffers";
    CUDA_ERROR_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    BOOST_LOG_TRIVIAL(debug) << "Free GPU memory: " << free_mem << " bytes";
    _accumulation_buffer.resize(_output_nchans, 0.0f);
    CUDA_ERROR_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    BOOST_LOG_TRIVIAL(debug) << "Free GPU memory after acc buffer: " << free_mem << " bytes";
    _h_accumulation_buffer.resize(_output_nchans, 0.0f);
    BOOST_LOG_TRIVIAL(debug) << "Allocating " << _chans_per_copy * _fft_length * 8  * 2 << " bytes for FFT buffers";
    _fft_input_buffer.resize(_chans_per_copy * _fft_length);
    _fft_output_buffer.resize(_chans_per_copy * _fft_length);
    CUDA_ERROR_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    BOOST_LOG_TRIVIAL(debug) << "Free GPU memory after FFT buffer: " << free_mem << " bytes";
    _copy_buffer.resize(_chans_per_copy * _fft_length);
    CUDA_ERROR_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    BOOST_LOG_TRIVIAL(debug) << "Free GPU memory after copy buffer: " << free_mem << " bytes";

    // Allocate streams
    BOOST_LOG_TRIVIAL(debug) << "Allocating CUDA streams";
    CUDA_ERROR_CHECK(cudaStreamCreate(&_copy_stream));
    CUDA_ERROR_CHECK(cudaStreamCreate(&_proc_stream));

    CUDA_ERROR_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    BOOST_LOG_TRIVIAL(debug) << "Free GPU memory pre-cufft: " << free_mem << " bytes";
    // Configure CUFFT for FFT execution

    BOOST_LOG_TRIVIAL(debug) << "Generating CUFFT plan";
    int n[] = {static_cast<int>(_fft_length)};
    int inembed[] = {static_cast<int>(_chans_per_copy)};
    int onembed[] = {static_cast<int>(_fft_length)};
    CUFFT_ERROR_CHECK(cufftPlanMany(&_fft_plan, 1, n, inembed, _chans_per_copy, 1, onembed, 1, _fft_length,
        CUFFT_C2C, _chans_per_copy));

    BOOST_LOG_TRIVIAL(debug) << "Setting CUFFT stream";
    CUFFT_ERROR_CHECK(cufftSetStream(_fft_plan, _proc_stream));
    CUDA_ERROR_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    BOOST_LOG_TRIVIAL(debug) << "Free GPU memory post-cufft: " << free_mem << " bytes";
    BOOST_LOG_TRIVIAL(debug) << "RSSpectrometer instance initialised";
}

RSSpectrometer::~RSSpectrometer()
{
    BOOST_LOG_TRIVIAL(debug) << "Destroying RSSpectrometer instance";
    BOOST_LOG_TRIVIAL(debug) << "Destroying CUDA streams";
    CUDA_ERROR_CHECK(cudaStreamDestroy(_copy_stream));
    CUDA_ERROR_CHECK(cudaStreamDestroy(_proc_stream));
    BOOST_LOG_TRIVIAL(debug) << "Destroying CUFFT plan";
    CUFFT_ERROR_CHECK(cufftDestroy(_fft_plan));
    BOOST_LOG_TRIVIAL(info) << "RSSpectrometer destroyed";
}

void RSSpectrometer::init(RawBytes &header)
{
    BOOST_LOG_TRIVIAL(debug) << "RSSpectrometer received header block";
}

bool RSSpectrometer::operator()(RawBytes &block)
{
    BOOST_LOG_TRIVIAL(debug) << "RSSpectrometer received data block";
    if (_nskip > 0)
    {
        BOOST_LOG_TRIVIAL(debug) << "Skipping block while stream stabilizes";
        --_nskip;
        return false;
    }
    assert(block.used_bytes() % _bytes_per_input_spectrum == 0); /** Block is not a multiple of the heap group size */
    std::size_t nspectra_in = block.used_bytes() / _bytes_per_input_spectrum;
    BOOST_LOG_TRIVIAL(debug) << "Number of input spectra per block: " << nspectra_in;
    assert(block.used_bytes() % _output_nchans * sizeof(InputType) == 0); /** Block is not a multiple of the spectrum size */
    std::size_t nspectra_out = block.used_bytes() / (_output_nchans * sizeof(InputType));
    BOOST_LOG_TRIVIAL(debug) << "Number of output spectra per block: " << nspectra_out;

    std::size_t n_to_accumulate;
    if (nspectra_out > _naccumulate)
    {
        n_to_accumulate = _naccumulate;
    }
    else
    {
        n_to_accumulate = nspectra_out;
    }
    BOOST_LOG_TRIVIAL(debug) << "Number of spectra to accumulate in current block: " << n_to_accumulate;
    BOOST_LOG_TRIVIAL(debug) << "Entering processing loop";
    std::size_t nchan_blocks = _input_nchans / _chans_per_copy;
    for (std::size_t spec_idx = 0; spec_idx < n_to_accumulate; ++spec_idx)
    {
        copy(block, spec_idx, 0, nspectra_in);
        for (std::size_t chan_block_idx = 1;
            chan_block_idx < nchan_blocks;
            ++chan_block_idx)
        {
            copy(block, spec_idx, chan_block_idx, nspectra_in);
            process(chan_block_idx - 1);
        }
        CUDA_ERROR_CHECK(cudaStreamSynchronize(_copy_stream));
        _copy_buffer.swap();
        process(nchan_blocks - 1);
    }
    CUDA_ERROR_CHECK(cudaStreamSynchronize(_proc_stream));
    BOOST_LOG_TRIVIAL(debug) << "Processing loop complete";
    _naccumulate -= n_to_accumulate;
    _naccumulated += n_to_accumulate;
    BOOST_LOG_TRIVIAL(info) << "Accumulated " << n_to_accumulate
    << " spectra ("<< _naccumulate << " remaining)";
    if (_naccumulate == 0)
    {
        BOOST_LOG_TRIVIAL(debug) << "Processing loop complete";
        // Here we need to do the final scaling and conversion
        thrust::transform(_accumulation_buffer.begin(), _accumulation_buffer.end(),
            _accumulation_buffer.begin(),
            kernels::convert_to_dBm(1000.0f / (FSW_IMPEDANCE * _naccumulated), 0));
        write_spectrum();
        // Free up some memory for histogram calculation
        _fft_output_buffer.resize(0);

        // Here we can calculate the histogram of the last block
        thrust::device_vector<int> d_hist;
        histogram(_fft_input_buffer, d_hist, 0.0, 2.0, 1024);
        write_histogram(d_hist);

        return true;
    }
    return false;
}

void RSSpectrometer::process(std::size_t chan_block_idx)
{
    /** Note streams do not actually work as expected
     *  with Thrust. The code is synchronous with respect
     *  to the host. The Thrust 1.9.4 (CUDA 10.1) release
     *  includes thrust::async which alleviates this problem.
     *  This can be included here if need be, but as it is the
     *  H2D copy should still run in parallel to the FFT, so
     *  there is no performance cost to blocking on the host.
     */
    CUDA_ERROR_CHECK(cudaStreamSynchronize(_proc_stream));
    // Convert shorts to floats
    BOOST_LOG_TRIVIAL(debug) << "Performing short2 to float2 conversion";

    thrust::transform(
        thrust::cuda::par.on(_proc_stream),
        _copy_buffer.b().begin(),
        _copy_buffer.b().end(),
        _fft_input_buffer.begin(),
        kernels::short2_be_to_float2_le());

    float scale_factor;
    if (_input_nchans == 1)
    {
        scale_factor = PASSTHROUGH_MODE_IQ_SCALING * sqrtf( powf(10.0f, 
				(_reference_dbm - 30.0f) / 10.0f) * 50.0f);
    }
    else if (_input_nchans == (1<<15))
    {
        scale_factor = PFB_MODE_IQ_SCALING;
    }
    else
    {
        BOOST_LOG_TRIVIAL(warning) << "No IQ scale factor known for " << _input_nchans << " channel input";
        scale_factor = 1.0f;
    }

    // Calculate RMS of data
    /*
    float sum = thrust::transform_reduce(
        thrust::cuda::par.on(_proc_stream),
        _fft_input_buffer.begin(),
        _fft_input_buffer.end(),
        kernels::detect_scale(scale_factor),
        0.0f,
        thrust::plus<float>());
    float rms = sqrtf(sum / _fft_input_buffer.size());
    BOOST_LOG_TRIVIAL(debug) << "RMS voltage of IQ data: " << rms << " V";
    */
    // Perform forward C2C transform
    BOOST_LOG_TRIVIAL(debug) << "Executing FFT";
    cufftComplex* in_ptr = static_cast<cufftComplex*>(
        thrust::raw_pointer_cast(_fft_input_buffer.data()));
    cufftComplex* out_ptr = static_cast<cufftComplex*>(
        thrust::raw_pointer_cast(_fft_output_buffer.data()));
    CUFFT_ERROR_CHECK(cufftExecC2C(
        _fft_plan, in_ptr, out_ptr, CUFFT_FORWARD));
    std::size_t chan_offset = chan_block_idx * _chans_per_copy * _fft_length;
    // Detect FFT output and accumulate
    BOOST_LOG_TRIVIAL(debug) << "Detecting and accumulating";

    thrust::transform(
        thrust::cuda::par.on(_proc_stream),
        _fft_output_buffer.begin(),
        _fft_output_buffer.end(),
        _accumulation_buffer.begin() + chan_offset,
        _accumulation_buffer.begin() + chan_offset,
        kernels::detect_accumulate(scale_factor/_fft_length));

}

void RSSpectrometer::copy(RawBytes& block, std::size_t spec_idx, std::size_t chan_block_idx, std::size_t nspectra_in)
{
    BOOST_LOG_TRIVIAL(debug) << "Copying block to GPU";
    std::size_t spitch = _input_nchans * sizeof(short2); // Width of a row in bytes (so number of channels total)
    std::size_t width = _chans_per_copy * sizeof(short2);; // Total number of samples in the input
    std::size_t dpitch = _chans_per_copy * sizeof(short2); // Width of row in bytes in the output
    std::size_t height = _fft_length; // Total number of samples to copy
    CUDA_ERROR_CHECK(cudaStreamSynchronize(_copy_stream));
    _copy_buffer.swap();

    if (_input_nchans != 1)
    {
        char* src = block.ptr() + spec_idx * spitch * height + chan_block_idx * width;
        BOOST_LOG_TRIVIAL(debug) << "Calling cudaMemcpy2DAsync with args: "
    	    << "dest=" << _copy_buffer.a_ptr() << ", "
    	    << "dpitch=" << dpitch << ", "
    	    << "src=" << (void*) src << ", "
    	    << "spitch=" << spitch << ", "
    	    << "width=" << width << ", "
    	    << "height=" << height << ", "
    	    << cudaMemcpyHostToDevice << ", "
    	    << _copy_stream;
        CUDA_ERROR_CHECK(cudaMemcpy2DAsync(_copy_buffer.a_ptr(),
            dpitch, src, spitch, width, height,
            cudaMemcpyHostToDevice, _copy_stream));
    }
    else
    {
        std::size_t nbytes = _fft_length * sizeof(short2);
        char* src = block.ptr() + spec_idx * nbytes;
        CUDA_ERROR_CHECK(cudaMemcpyAsync(_copy_buffer.a_ptr(), src, nbytes,
            cudaMemcpyHostToDevice, _copy_stream));
    }
}

void RSSpectrometer::write_histogram(thrust::device_vector<int> const& histogram)
{
    // Copy histogeam buffer to host
    BOOST_LOG_TRIVIAL(debug) << "Copying histogram to host";
    thrust::host_vector<int> h_hist = histogram;
    BOOST_LOG_TRIVIAL(debug) << "Perparing output file";
    std::ofstream outfile;
    std::string _hist_filename(_filename + ".hist");
    outfile.open(_hist_filename.c_str(),std::ifstream::out | std::ifstream::binary);
    if (outfile.is_open())
    {
        BOOST_LOG_TRIVIAL(debug) << "Opened file " << _hist_filename;
    }
    else
    {
        std::stringstream stream;
        stream << "Could not open file " << _hist_filename;
        throw std::runtime_error(stream.str().c_str());
    }
    outfile.write((char*)h_hist.data(), h_hist.size() * sizeof(int));
    outfile.flush();
    outfile.close();
}

void RSSpectrometer::write_spectrum()
{
    // Copy accumulation buffer to host
    BOOST_LOG_TRIVIAL(debug) << "Copying accumulated spectrum to host";
    _h_accumulation_buffer = _accumulation_buffer;
    BOOST_LOG_TRIVIAL(debug) << "Perparing output file";
    std::ofstream outfile;
    outfile.open(_filename.c_str(),std::ifstream::out | std::ifstream::binary);
    if (outfile.is_open())
    {
        BOOST_LOG_TRIVIAL(debug) << "Opened file " << _filename;
    }
    else
    {
        std::stringstream stream;
        stream << "Could not open file " << _filename;
        throw std::runtime_error(stream.str().c_str());
    }
    BOOST_LOG_TRIVIAL(info) << "Writing output to " << _filename << " with FFT shifts included";
    // Here we are now doing a double FFT shift
    // We must first shift the contents of every coarse channel
    // The we write out the full spectrum with a shift
    // First write second half of the band
    std::size_t nsubbands = _h_accumulation_buffer.size() / _fft_length;
    for (std::size_t subband_idx=nsubbands/2; subband_idx < nsubbands; ++subband_idx)
    {
        std::size_t offset = subband_idx * _fft_length;
        //First write upper half of the band
        char* back = reinterpret_cast<char*>(&_h_accumulation_buffer[offset + _fft_length/2]);
        char* front = reinterpret_cast<char*>(&_h_accumulation_buffer[offset]);
        outfile.write(back, (_fft_length/2) * sizeof(decltype(_h_accumulation_buffer)::value_type));
        outfile.write(front, (_fft_length/2) * sizeof(decltype(_h_accumulation_buffer)::value_type));
    }
    // Second write out the first half of the band
    for (std::size_t subband_idx=0; subband_idx < nsubbands/2; ++subband_idx)
    {
        std::size_t offset = subband_idx * _fft_length;
        //First write upper half of the band
        char* back = reinterpret_cast<char*>(&_h_accumulation_buffer[offset + _fft_length/2]);
        char* front = reinterpret_cast<char*>(&_h_accumulation_buffer[offset]);
        outfile.write(back, (_fft_length/2) * sizeof(decltype(_h_accumulation_buffer)::value_type));
        outfile.write(front, (_fft_length/2) * sizeof(decltype(_h_accumulation_buffer)::value_type));
    }
    outfile.flush();
    outfile.close();
    BOOST_LOG_TRIVIAL(debug) << "File write complete";
}


} //namespace rfi_chamber
} //namespace effelsberg
} //namespace psrdada_cpp

