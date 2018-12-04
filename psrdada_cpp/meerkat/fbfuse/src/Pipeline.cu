#include "psrdada_cpp/meerkat/fbfuse/Pipeline.cuh"
#include "psrdada_cpp/meerkat/fbfuse/Header.hpp"
#include "psrdada_cpp/cuda_utils.hpp"
#include "ascii_header.h"
#include "cuda.h"
#include <stdexcept>
#include <exception>
#include <cstdlib>

#define FBFUSE_SAMPLE_CLOCK_START_KEY "SAMPLE_CLOCK_START"
#define FBFUSE_SAMPLE_CLOCK_KEY "SAMPLE_CLOCK"
#define FBFUSE_SYNC_TIME_KEY "SYNC_TIME"

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {

Pipeline::Pipeline(PipelineConfig const& config,
    DadaWriteClient& cb_writer,
    DadaWriteClient& ib_writer,
    std::size_t input_data_buffer_size)
    : _config(config)
    , _sample_clock_start(0)
    , _sample_clock(0.0)
    , _sync_time(0.0)
    , _unix_timestamp(0.0)
    , _sample_clock_tick_per_block(0)
    , _call_count(0)
    , _cb_writer(cb_writer)
    , _cb_header_stream(cb_writer.header_stream())
    , _cb_data_stream(cb_writer.data_stream())
    , _ib_writer(ib_writer)
    , _ib_header_stream(ib_writer.header_stream())
    , _ib_data_stream(ib_writer.data_stream())
{
    BOOST_LOG_TRIVIAL(debug) << "Verifying all DADA buffer capacities";
    // Here we should check the size of all the input and output
    // and throw an error on incorrect buffer sizes.
    //
    // Input buffer checks:
    //
    std::size_t heap_group_size = (_config.total_nantennas() * _config.nchans()
        * _config.nsamples_per_heap() * _config.npol()) * sizeof(char2);

    if (input_data_buffer_size % heap_group_size != 0)
    {
        throw std::runtime_error("Input DADA buffer is not a multiple "
            "of the expected heap group size");
    }
    _nheap_groups_per_block = input_data_buffer_size / heap_group_size;
    _nsamples_per_dada_block = _nheap_groups_per_block * _config.nsamples_per_heap();
    BOOST_LOG_TRIVIAL(debug) << "Number of heap groups per block: " << _nheap_groups_per_block;
    BOOST_LOG_TRIVIAL(debug) << "Number of samples/spectra per block: " << _nsamples_per_dada_block;
    if (_nsamples_per_dada_block % _config.cb_nsamples_per_block() != 0)
    {
        throw std::runtime_error("Input DADA buffer does not contain an integer "
            "multiple of the required number of samples per device block");
    }
    _taftp_db.resize(input_data_buffer_size / sizeof(char2), {0,0});

    //
    // Output buffer checks:
    //
    std::size_t expected_cb_size = (_config.cb_nbeams() * _nsamples_per_dada_block
        / _config.cb_tscrunch() * _config.nchans() / _config.cb_fscrunch()) * sizeof(int8_t);
    if (_cb_writer.data_buffer_size() != expected_cb_size)
    {
        throw std::runtime_error(
            std::string("Expected coherent beam output buffer to have a size of ")
            + std::to_string(expected_cb_size)
            + " bytes, but it instead had a size of "
            + std::to_string(_cb_writer.data_buffer_size())
            + " bytes");
    }
    _tbtf_db.resize(expected_cb_size, 0);

    std::size_t expected_ib_size = (_config.ib_nbeams() * _nsamples_per_dada_block
        / _config.ib_tscrunch() * _config.nchans() / _config.ib_fscrunch()) * sizeof(int8_t);
    if (_ib_writer.data_buffer_size() != expected_ib_size)
    {
        throw std::runtime_error(
            std::string("Expected incoherent beam output buffer to have a size of ")
            + std::to_string(expected_ib_size)
            + " bytes, but it instead had a size of "
            + std::to_string(_ib_writer.data_buffer_size())
            + " bytes");
    }
    _tf_db.resize(expected_ib_size, 0);

    // Calculate the timestamp step per block
    _sample_clock_tick_per_block = 2 * _config.total_nchans() * _nsamples_per_dada_block;
    BOOST_LOG_TRIVIAL(debug) << "Sample clock tick per block: " << _sample_clock_tick_per_block;

    BOOST_LOG_TRIVIAL(debug) << "Allocating CUDA streams";
    CUDA_ERROR_CHECK(cudaStreamCreate(&_h2d_copy_stream));
    CUDA_ERROR_CHECK(cudaStreamCreate(&_processing_stream));
    CUDA_ERROR_CHECK(cudaStreamCreate(&_d2h_copy_stream));

    BOOST_LOG_TRIVIAL(debug) << "Constructing delay and weights managers";
    _delay_manager.reset(new DelayManager(_config, _h2d_copy_stream));
    _weights_manager.reset(new WeightsManager(_config, _processing_stream));
    _split_transpose.reset(new SplitTranspose(_config));
    _coherent_beamformer.reset(new CoherentBeamformer(_config));
    _incoherent_beamformer.reset(new IncoherentBeamformer(_config));
}

Pipeline::~Pipeline()
{
    try
    {
        _cb_data_stream.release();
        _ib_data_stream.release();
    }
    catch (std::exception& e)
    {
        BOOST_LOG_TRIVIAL(warning) << "Non-fatal error on pipeline destruction: "
        << e.what();
    }
    CUDA_ERROR_CHECK(cudaStreamDestroy(_h2d_copy_stream));
    CUDA_ERROR_CHECK(cudaStreamDestroy(_processing_stream));
    CUDA_ERROR_CHECK(cudaStreamDestroy(_d2h_copy_stream));
}

void Pipeline::set_header(RawBytes& header)
{
    Header parser(header);
    parser.purge();
    // There is a bug in DADA that results in keys made of subkeys not being writen if
    // the superkey is writen first. To get around this the order of key writes needs to
    // be carefully considered.
    parser.set<long double>(FBFUSE_SAMPLE_CLOCK_KEY, _sample_clock);
    parser.set<long double>(FBFUSE_SYNC_TIME_KEY, _sync_time);
    parser.set<std::size_t>(FBFUSE_SAMPLE_CLOCK_START_KEY, _sample_clock_start);
    header.used_bytes(header.total_bytes());
}

void Pipeline::init(RawBytes& header)
{
    BOOST_LOG_TRIVIAL(debug) << "Parsing DADA header";
    // Extract the time from the header and convert it to a double epoch
    Header parser(header);
    _sample_clock_start = parser.get<std::size_t>(FBFUSE_SAMPLE_CLOCK_START_KEY);
    _sample_clock = parser.get<long double>(FBFUSE_SAMPLE_CLOCK_KEY);
    _sync_time = parser.get<long double>(FBFUSE_SYNC_TIME_KEY);

    // Need to set the header information on the coherent beam output block
    auto& cb_header_block = _cb_header_stream.next();
    set_header(cb_header_block);
    _cb_header_stream.release();

    // Need to set the header information on the incoherent beam output block
    auto& ib_header_block = _ib_header_stream.next();
    set_header(ib_header_block);
    _ib_header_stream.release();
}

void Pipeline::process(VoltageVectorType const& taftp_vec,
    PowerVectorType& tbtf_vec, PowerVectorType& tf_vec)
{
    BOOST_LOG_TRIVIAL(debug) << "Executing coherent beamforming pipeline";
    BOOST_LOG_TRIVIAL(debug) << "Checking for delay updates";
    auto const& delays = _delay_manager->delays();
    BOOST_LOG_TRIVIAL(debug) << "Calculating weights at unix time: " << _unix_timestamp;
    auto const& weights = _weights_manager->weights(delays, _unix_timestamp);
    BOOST_LOG_TRIVIAL(debug) << "Transposing input data from TAFTP to FTPA order";
    _split_transpose->transpose(taftp_vec, _split_transpose_output, _processing_stream);
    BOOST_LOG_TRIVIAL(debug) << "Forming coherent beams";
    _coherent_beamformer->beamform(_split_transpose_output, weights, tbtf_vec, _processing_stream);
    BOOST_LOG_TRIVIAL(debug) << "Forming incoherent beam";
    _incoherent_beamformer->beamform(taftp_vec, tf_vec, _processing_stream);
}

bool Pipeline::operator()(RawBytes& data)
{
    // This update goes at the top of the method to ensure
    // that it is always incremented on each pass
    ++_call_count;

    BOOST_LOG_TRIVIAL(debug) << "Pipeline::operator() called (count = " << _call_count << ")";
    // We first need to synchronize the h2d copy stream to ensure that
    // last host to device copy has completed successfully. When this is
    // done we are free to call swap on the double buffer without affecting
    // any previous copy.
    CUDA_ERROR_CHECK(cudaStreamSynchronize(_h2d_copy_stream));
    _taftp_db.swap();

    if (data.used_bytes() != _taftp_db.a().size()*sizeof(char2))
    {

        throw std::runtime_error(std::string("Unexpected DADA buffer size, expected ")
            + std::to_string(_taftp_db.a().size()*sizeof(char2))
            + " but got "
            + std::to_string(data.used_bytes()));
    }
    CUDA_ERROR_CHECK(cudaMemcpyAsync(static_cast<void*>(_taftp_db.a_ptr()),
        static_cast<void*>(data.ptr()), data.used_bytes(),
        cudaMemcpyHostToDevice, _h2d_copy_stream));


    // If we are on the first call we can exit here as there is no
    // data on the GPU yet to process.
    if (_call_count == 1)
    {
        return false;
    }
    // Here we block on the processing stream before swapping
    // the processing buffers
    CUDA_ERROR_CHECK(cudaStreamSynchronize(_processing_stream));
    _tbtf_db.swap();
    _tf_db.swap();
    // Calculate the unix timestamp for the block that is about to be processed
    // (which is the block passed the last time that operator() was called)
    _unix_timestamp = (_sync_time + (_sample_clock_start +
        ((_call_count - 2) * _sample_clock_tick_per_block))
    / _sample_clock);
    process(_taftp_db.b(), _tbtf_db.a(), _tf_db.a());

    // If we are on the second call we can exit here as there is not data
    // that has completed processing at this stage.
    if (_call_count == 2)
    {
        return false;
    }

    CUDA_ERROR_CHECK(cudaStreamSynchronize(_d2h_copy_stream));
    // Only want to perform one copy per data block here, not d2h then h2h.
    // For this reason we need access to two DadaWriteClient instances in
    // this class.
    if (_call_count > 3)
    {
        // If the call count is >3 then we have already performed the first
        // output copy and we need to release and get the next dada blocks
        _cb_data_stream.release();
        _ib_data_stream.release();
    }
    auto& cb_block = _cb_data_stream.next();
    auto& ib_block = _ib_data_stream.next();
    CUDA_ERROR_CHECK(cudaMemcpyAsync(static_cast<void*>(cb_block.ptr()),
        static_cast<void*>(_tbtf_db.b_ptr()), cb_block.total_bytes(),
        cudaMemcpyDeviceToHost, _d2h_copy_stream));
    CUDA_ERROR_CHECK(cudaMemcpyAsync(static_cast<void*>(ib_block.ptr()),
        static_cast<void*>(_tf_db.b_ptr()), ib_block.total_bytes(),
        cudaMemcpyDeviceToHost, _d2h_copy_stream));
    cb_block.used_bytes(cb_block.total_bytes());
    ib_block.used_bytes(ib_block.total_bytes());
    return false;
}

} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp
