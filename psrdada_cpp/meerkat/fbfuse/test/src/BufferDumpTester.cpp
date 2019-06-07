
#include "psrdada_cpp/meerkat/fbfuse/test/BufferDumpTester.hpp"
#include "psrdada_cpp/dada_db.hpp"

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace test {

BufferDumpTester::BufferDumpTester()
    : ::testing::Test()
{

}

BufferDumpTester::~BufferDumpTester()
{
}

void BufferDumpTester::SetUp()
{
}

void BufferDumpTester::TearDown()
{
}

TEST_F(BufferDumpTester, do_nothing)
{

    std::size_t nchans = 64;
    std::size_t total_nchans = 4096;
    std::size_t nantennas = 64;
    std::size_t ngroups = 8;
    std::size_t nblocks = 64;
    std::size_t block_size = nchans * nantennas * ngroups * 256 * sizeof(unsigned);

    float cfreq = 856e6;
    float bw = 856e6 / (total_nchans / nchans);
    float max_fill_level = 80.0;

    DadaDB buffer(nblocks, block_size, 4, 4096);
    MultiLog log("log");
    DadaOutputStream ostream(buffer.dada_key(), log);

    std::vector<char> input_header_buffer(4096, 0);
    RawBytes input_header_rb(input_header_buffer.data(), 4096, 4096);
    Header header(input_header_rb);
    header.set<long double>("SAMPLE_CLOCK", 856000000.0);
    header.set<long double>("SYNC_TIME", 0.0);
    header.set<std::size_t>("SAMPLE_CLOCK_START", 0);
    ostream.init(input_header_rb);
    std::vector<char> input_data_buffer(block_size, 0);
    RawBytes input_data_rb(input_data_buffer, block_size, block_size);
    for (int ii=0; ii < nblocks-2; ++ii)
    {
        ostream(input_data_rb);
    }

    NullSink sink;
    DadaReadClient reader(buffer.dada_key(), log);
    BufferDump<decltype(sink)> dumper(reader, sink, max_fill_level, nantennas, nchans, total_nchans, cfreq, bw);

    std::thread dumper_thread([&](){
        dumper.start();
    });

    std::this_thread::sleep_for(std::chrono::seconds(1));

    dumper.stop();
}

} //namespace test
} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

