#include "psrdada_cpp/test/test_sigproc_file_writer.hpp"
#include "psrdada_cpp/sigproc_file_writer.hpp"
#include <sys/stat.h>

namespace psrdada_cpp {
namespace test {

TestSigprocFileWriter::TestSigprocFileWriter()
    : ::testing::Test()
{
}

TestSigprocFileWriter::~TestSigprocFileWriter()
{
}

void TestSigprocFileWriter::SetUp()
{
}

void TestSigprocFileWriter::TearDown()
{
}

void populate_header(FilHead& header)
{
    header.rawfile = "test.fil";
    header.source = "J0000+0000";
    header.az = 0.0;
    header.dec = 0.0;
    header.fch1 = 1400.0;
    header.foff = -0.03;
    header.ra = 0.0;
    header.rdm = 0.0;
    header.tsamp = 0.000064;
    header.tstart = 58758.0; //corresponds to 2019-10-02-00:00:00
    header.za = 0.0;
    header.datatype = 1;
    header.barycentric = 0;
    header.ibeam = 1;
    header.machineid = 0;
    header.nbeams = 1;
    header.nbits = 8;
    header.nchans = 1024;
    header.nifs = 1;
    header.telescopeid = 1;
}

bool is_file_valid(std::string const& name, std::size_t expected_size)
{
    struct stat buffer;
    int retval = stat(name.c_str(), &buffer);
    if (retval != 0)
    {
        std::cerr << name << " not accessible" << std::endl;
        return false;
    }
    else
    {
        return (std::size_t(buffer.st_size) == expected_size);
    }
}

TEST_F(TestSigprocFileWriter, test_filesize)
/* Test whether the files that are written are of the correct size */
{

    std::size_t nblocks = 5;
    std::size_t block_size = 2000;
    std::size_t desired_size = 1<<12;
    FilHead header;
    populate_header(header);

    SigprocFileWriter writer;
    writer.enable();
    writer.max_filesize(desired_size);
    writer.tag("test");
    writer.directory("/tmp");

    char* header_ptr = new char[4096];
    RawBytes header_block(header_ptr, 4096, 4096);

    char* data_ptr = new char[block_size];
    RawBytes data_block(data_ptr, block_size, block_size);

    SigprocHeader parser;
    std::size_t header_size = parser.write_header(header_ptr, header);

    writer.init(header_block);
    for (std::size_t ii = 0; ii < nblocks; ++ii)
    {
        writer(data_block);
    }

    writer.disable();
    writer(data_block);

    ASSERT_TRUE(is_file_valid("/tmp/2019-10-02-00:00:00_test_0.fil", desired_size + header_size));
    ASSERT_TRUE(is_file_valid("/tmp/2019-10-02-00:00:00_test_4096.fil", desired_size + header_size));
    ASSERT_TRUE(is_file_valid("/tmp/2019-10-02-00:00:00_test_8192.fil", 1808 + header_size));

    delete[] header_ptr;
    delete[] data_ptr;
}



} //namespace test
} //namespace psrdada_cpp

