#include "ascii_header.h"
#include <stdexcept>
#include <exception>
#include <cstdlib>

#define SAMPLE_CLOCK_START_KEY "SAMPLE_CLOCK_START"

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {


Pipeline::Pipeline(PipelineConfig const& config,
    DadaWriteClient& cb_writer,
    DadaWriteClient& ib_writer)
    : _config(config)
    , _start_epoch(0)
    , _call_count(0)
    , _cb_writer(cb_writer),
    , _cb_header_stream(cb_writer.header_stream())
    , _cb_data_stream(cb_writer.data_stream())
    , _ib_writer(ib_writer),
    , _ib_header_stream(ib_writer.header_stream())
    , _ib_data_stream(ib_writer.data_stream())
{
    CUDA_SAFE_CALL(cudaStreamCreate(&_h2d_copy_stream));
    CUDA_SAFE_CALL(cudaStreamCreate(&_processing_stream));
    CUDA_SAFE_CALL(cudaStreamCreate(&_d2h_copy_stream));
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
        BOOST_LOG_TRIVIAL(warn) << "Non-fatal error on pipeline destruction: "
        << e.what();
    }
    CUDA_SAFE_CALL(cudaStreamDestroy(_h2d_copy_stream));
    CUDA_SAFE_CALL(cudaStreamDestroy(_processing_stream));
    CUDA_SAFE_CALL(cudaStreamDestroy(_d2h_copy_stream));
}

void Pipeline::set_header(RawBlock& header)
{
    std::memset(static_cast<void*>(header.ptr()), 0, header.total_bytes());
    header.used_bytes(header.total_bytes());
    ascii_header_set(header.ptr(), SAMPLE_CLOCK_START_KEY, "%ul", _start_epoch);
}

void Pipeline::init(RawBlock& header)
{
    BOOST_LOG_TRIVIAL(debug) << "Parsing DADA header";
    // Extract the time from the header and convert it to a double epoch
    char tmp[32];
    if (ascii_header_get(header.ptr(), SAMPLE_CLOCK_START_KEY, "%s", tmp) == -1)
    {
        throw std::runtime_error("Could not fine SAMPLE_CLOCK_START key in DADA header.")
    }
    _start_epoch = std::strtoul(tmp, NULL, 0);
    BOOST_LOG_TRIVIAL(info) << SAMPLE_CLOCK_START_KEY << " = " << _start_epoch;

    // Need to set the header information on the coherent beam output block
    auto& cb_header_block = _cb_header_stream.next();
    set_header(cb_header_block);
    _cb_header_stream.release();

    // Need to set the header information on the incoherent beam output block
    auto& ib_header_block = _ib_header_stream.next();
    set_header(ib_header_block);
    _ib_header_stream.release();
}

void Pipeline::process(char2* taftp_ptr, char* tbftf_ptr, char* tftf_ptr)
{

}

bool Pipeline::operator()(RawBlock& data)
{
    ++_call_count;
    if (_call_count == 1)
    {
        // Here we should check the size of all the input and output
        // and throw an error on incorrect buffer sizes.
        //
        // Input buffer checks:
        //
        std::size_t heap_group_size = (FBFUSE_TOTAL_ANTENNAS * FBFUSE_NCHANS
            * FBFUSE_NSAMPLES_PER_HEAP * FBFUSE_NPOL) * sizeof(char2);
        if (data.used_bytes() % heap_group_size != 0)
        {
            throw std::runtime_error("Input DADA buffer is not a multiple "
                "of the expected heap group size");
        }
        _nheap_groups_per_block = data.used_bytes() / heap_group_size;
        _nsamples_per_dada_block = _nheap_groups_per_block * FBFUSE_NSAMPLES_PER_HEAP;
        if (_nsamples_per_dada_block % FBFUSE_CB_NSAMPLES_PER_BLOCK != 0)
        {
            throw std::runtime_error("Input DADA buffer does not contain an integer "
                "multiple of the required number of samples per device block");
        }
        _taftp_db.resize(heap_group_size / sizeof(char2), 0);

        //
        // Output buffer checks:
        //
        std::size_t expected_cb_size = (FBFUSE_CB_NBEAMS * _nsamples_per_dada_block
            / FBFUSE_CB_TSCRUNCH * FBFUSE_NCHANS / FBFUSE_CB_FSCRUNCH) * sizeof(char);
        if (_cb_writer.data_buffer_size() != expected_cb_size)
        {
            throw std::runtime_error(
                std::string("Expected coherent beam output buffer to have a size of ")
                + std::to_string(expected_cb_size)
                + " bytes, but it instead had a size of "
                + std::to_string(_cb_writer.data_buffer_size())
                + " bytes");
        }
        _tbftf_db.resize(expected_cb_size, 0);

        std::size_t expected_ib_size = (FBFUSE_IB_NBEAMS * _nsamples_per_dada_block
            / FBFUSE_IB_TSCRUNCH * FBFUSE_NCHANS / FBFUSE_IB_FSCRUNCH) * sizeof(char);
        if (_ib_writer.data_buffer_size() != expected_ib_size)
        {
            throw std::runtime_error(
                std::string("Expected incoherent beam output buffer to have a size of ")
                + std::to_string(expected_ib_size)
                + " bytes, but it instead had a size of "
                + std::to_string(_ib_writer.data_buffer_size())
                + " bytes");
        }
        _tftf_db.resize(expected_ib_size, 0);
    }

    // We first need to synchronize the h2d copy stream to ensure that
    // last host to device copy has completed successfully. When this is
    // done we are free to call swap on the double buffer without affecting
    // any previous copy.
    CUDA_SAFE_CALL(cudaStreamSynchronize(_h2d_copy_stream));
    _taftp_db.swap();
    CUDA_SAFE_CALL(cudaMemcpyAsync(static_cast<void*>(_taftp_db.a()),
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
    CUDA_SAFE_CALL(cudaStreamSynchronize(_processing_stream));
    _tbftf_db.swap();
    _tftf_db.swap();
    process(_taftp_db.b(), _tbftf_db.a(), _tftf_db.a());

    // If we are on the second call we can exit here as there is not data
    // that has completed processing at this stage.
    if (_call_count == 2)
    {
        return false;
    }

    CUDA_SAFE_CALL(cudaStreamSynchronize(_d2h_copy_stream));
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
    CUDA_SAFE_CALL(cudaMemcpyAsync(static_cast<void*>(cb_block.ptr()),
        static_cast<void*>(_tbftf_db.b()), cb_block.total_bytes(),
        cudaMemcpyDeviceToHost, _d2h_copy_stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(static_cast<void*>(ib_block.ptr()),
        static_cast<void*>(_tftf_db.b()), ib_block.total_bytes(),
        cudaMemcpyDeviceToHost, _d2h_copy_stream));

    return false;
}

} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

unsigned long int strtoul (const char* str, char** endptr, int base);