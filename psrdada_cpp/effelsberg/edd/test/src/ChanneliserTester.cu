#include "psrdada_cpp/effelsberg/edd/test/ChanneliserTester.cuh"
#include "psrdada_cpp/effelsberg/edd/Channeliser.cuh"
#include "psrdada_cpp/dada_null_sink.hpp"
#include "psrdada_cpp/dada_db.hpp"
#include "psrdada_cpp/dada_write_client.hpp"
#include "psrdada_cpp/dada_input_stream.hpp"
#include "psrdada_cpp/double_host_buffer.cuh"
#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/cuda_utils.hpp"
#include <vector>
#include <thread>
#include <chrono>
#include <exception>

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {
namespace test {

ChanneliserTester::ChanneliserTester()
    : ::testing::Test()
{

}

ChanneliserTester::~ChanneliserTester()
{

}

void ChanneliserTester::SetUp()
{
}

void ChanneliserTester::TearDown()
{
}

void ChanneliserTester::performance_test(std::size_t fft_length, std::size_t nbits)
{
    std::size_t input_block_bytes = fft_length * 8192 * 1024 * nbits / 8;
    std::size_t output_block_bytes = (fft_length/2 + 1) * 8192 * 1024 * sizeof(Channeliser::PackedChannelisedVoltageType);
    DadaDB output_buffer(8, output_block_bytes);
    output_buffer.create();
    MultiLog log("test_log");
    NullSink null_sink;
    DadaInputStream<NullSink> consumer(output_buffer.key(), log, null_sink);
    std::thread consumer_thread( [&](){
        try {
            consumer.start();
        } catch (std::exception& e) {
            BOOST_LOG_TRIVIAL(error) << e.what();
        }
    });
    DadaWriteClient client(output_buffer.key(), log);
    DoublePinnedHostBuffer<char> input_block;
    input_block.resize(input_block_bytes);	
    RawBytes input_raw_bytes(input_block.a_ptr(), input_block_bytes, input_block_bytes);
    std::vector<char> header_block(4096);
    RawBytes header_raw_bytes(header_block.data(), 4096, 4096);
    Channeliser channeliser(input_block_bytes, fft_length, nbits, 16.0f, 16.0f, client);
    channeliser.init(header_raw_bytes);
    for (int ii = 0; ii < 100; ++ii)
    {
        channeliser(input_raw_bytes);
    }
    consumer.stop();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    channeliser(input_raw_bytes);
    consumer_thread.join();
}


TEST_F(ChanneliserTester, simple_exec_test)
{
    performance_test(16, 12);
    performance_test(32, 8);
}

} //namespace test
} //namespace edd
} //namespace meerkat
} //namespace psrdada_cpp
