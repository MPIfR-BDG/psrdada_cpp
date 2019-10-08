#include "psrdada_cpp/meerkat/fbfuse/test/PipelineTester.cuh"
#include "psrdada_cpp/meerkat/fbfuse/fbfuse_constants.hpp"
#include "psrdada_cpp/meerkat/fbfuse/PipelineConfig.hpp"
#include "psrdada_cpp/meerkat/fbfuse/DelayEngineSimulator.cuh"
#include "psrdada_cpp/Header.hpp"
#include "psrdada_cpp/dada_null_sink.hpp"
#include "psrdada_cpp/dada_read_client.hpp"
#include "psrdada_cpp/dada_input_stream.hpp"
#include "psrdada_cpp/dada_db.hpp"
#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/cuda_utils.hpp"
#include <random>
#include <cmath>
#include <complex>
#include <thread>
#include <chrono>
#include <vector>
#include <exception>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace test {

PipelineTester::PipelineTester()
    : ::testing::Test()
{
}

PipelineTester::~PipelineTester()
{
}

void PipelineTester::SetUp()
{
    _config.centre_frequency(1.4e9);
    _config.bandwidth(56.0e6);
    _config.delay_buffer_shm("test_delay_buffer_shm");
    _config.delay_buffer_sem("test_delay_buffer_sem");
    _config.delay_buffer_mutex("test_delay_buffer_mutex");
}

void PipelineTester::TearDown()
{
}

TEST_F(PipelineTester, simple_run_test)
{

    DelayEngineSimulator simulator(_config);

    int const ntimestamps_per_block = 64;
    int const taftp_block_size = (ntimestamps_per_block * _config.total_nantennas()
        * _config.nchans() * _config.nsamples_per_heap() * _config.npol());
    int const taftp_block_bytes = taftp_block_size * sizeof(char2);

    //Create output buffer for coherent beams
    int const cb_output_nsamps = _config.nsamples_per_heap() * ntimestamps_per_block / _config.cb_tscrunch();
    int const cb_output_nchans = _config.nchans() / _config.cb_fscrunch();
    int const cb_block_size = _config.cb_nbeams() * cb_output_nsamps * cb_output_nchans;
    DadaDB cb_buffer(8, cb_block_size, 4, 4096);
    cb_buffer.create();
    _config.cb_dada_key(cb_buffer.key());

    //Create output buffer for incoherent beams
    int const ib_output_nsamps = _config.nsamples_per_heap() * ntimestamps_per_block / _config.ib_tscrunch();
    int const ib_output_nchans = _config.nchans() / _config.ib_fscrunch();
    int const ib_block_size = _config.ib_nbeams() * ib_output_nsamps * ib_output_nchans;
    DadaDB ib_buffer(8, ib_block_size, 4, 4096);
    ib_buffer.create();
    _config.ib_dada_key(ib_buffer.key());

    //Setup write clients
    MultiLog log("PipelineTester");
    DadaWriteClient cb_write_client(_config.cb_dada_key(), log);
    DadaWriteClient ib_write_client(_config.ib_dada_key(), log);
    Pipeline pipeline(_config, cb_write_client, ib_write_client, taftp_block_bytes);

    //Set up null sinks on all buffers
    NullSink null_sink;

    DadaInputStream<NullSink> cb_consumer(_config.cb_dada_key(), log, null_sink);
    DadaInputStream<NullSink> ib_consumer(_config.ib_dada_key(), log, null_sink);

    std::thread cb_consumer_thread( [&](){
        try {
            cb_consumer.start();
        } catch (std::exception& e) {
            BOOST_LOG_TRIVIAL(error) << e.what();
        }
	});
    std::thread ib_consumer_thread( [&](){
        try {
            ib_consumer.start();
        } catch (std::exception& e) {
            BOOST_LOG_TRIVIAL(error) << e.what();
        }
	});

    //Create and input header buffer
    std::vector<char> input_header_buffer(4096, 0);
    RawBytes input_header_rb(input_header_buffer.data(), 4096, 4096);
    Header header(input_header_rb);
    header.set<long double>("SAMPLE_CLOCK", 856000000.0);
    header.set<long double>("SYNC_TIME", 0.0);
    header.set<std::size_t>("SAMPLE_CLOCK_START", 0);

    //Create and input data buffer
    char* input_data_buffer;
    CUDA_ERROR_CHECK(cudaMallocHost((void**)&input_data_buffer, taftp_block_bytes));
    RawBytes input_data_rb(input_data_buffer, taftp_block_bytes, taftp_block_bytes);

    //Run the init
    pipeline.init(input_header_rb);
    //Loop over N data blocks and push them through the system
    for (int ii = 0; ii < 200; ++ii)
    {
        pipeline(input_data_rb);
    }
    cb_consumer.stop();
    ib_consumer.stop();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    pipeline(input_data_rb);
    cb_consumer_thread.join();
    ib_consumer_thread.join();
    CUDA_ERROR_CHECK(cudaFreeHost((void*)input_data_buffer));
}

} //namespace test
} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

