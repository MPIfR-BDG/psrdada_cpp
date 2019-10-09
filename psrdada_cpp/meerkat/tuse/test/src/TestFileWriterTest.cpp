#include "psrdada_cpp/meerkat/tuse/test/TestFileWriterTest.h"
#include "psrdada_cpp/sigprocheader.hpp"
#include "psrdada_cpp/dada_output_stream.hpp"
#include "psrdada_cpp/dada_input_stream.hpp"
#include "psrdada_cpp/dada_db.hpp"
#include "psrdada_cpp/common.hpp"
#include <glob.h>
#include <cmath>
#include <chrono>
#include <vector>
#include <exception>


namespace psrdada_cpp {
namespace meerkat {
namespace tuse {
namespace test {

TestFileWriterTest::TestFileWriterTest()
    : ::testing::Test()
{
}

TestFileWriterTest::~TestFileWriterTest()
{
}

void TestFileWriterTest::SetUp()
{
}

void TestFileWriterTest::TearDown()
{
}

TEST_F(TestFileWriterTest, test_filesize)
/* Test whether the files that are written are of the correct size */
{
/* Setup  the Dada buffers and write to them */
    DadaDB dada_buffer(8, 10240, 4, 4096);
    dada_buffer.create();

    {
    //Setup write clients
    MultiLog log("Output buffer");
    DadaOutputStream outstream(dada_buffer.key(), log);

    // Setup RawBytes to write
    char* hdr_ptr = new char[4096];
    std::memset(hdr_ptr, 0, 4096);
    RawBytes header(hdr_ptr, 4096, 4096, false);
    std::ifstream file("header.txt");
    std::string str((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    const char* hdrstr = str.c_str();
    //const char* hdrstr = "HEADER_START and HEADER_END";
    std::memcpy(hdr_ptr, hdrstr,277);

    file.close();
    char* data_ptr = new char[10240];
    RawBytes data(data_ptr, 10240, 10240, false);
    // Write data to the buffer
    outstream.init(header);
    for (std::uint32_t ii =0 ; ii < 8; ++ii)
    {
        outstream(data);
    }


    // Write data to file
    MultiLog log1("input stream");
    std::string filename("Test_file");
    TestFileWriter testfiles(filename,15360);

    DadaReadClient client(dada_buffer.key(), log1);
    auto& header_stream = client.header_stream();
    auto& header_block = header_stream.next();
    testfiles.init(header_block);
    header_stream.release();

    // Read Client
    for (std::uint8_t jj = 0; jj < 8 ; ++jj )
    {
        auto& data_stream = client.data_stream();
        if (data_stream.at_end())
        {
            BOOST_LOG_TRIVIAL(info) << "Reached end of data";
            break;
        }
        auto& data_block = data_stream.next();
        testfiles(data_block);
        data_stream.release();
    }

    // Check for size
    for (std::uint8_t jj =0; jj < 3; ++jj)
    {
        std::ifstream fstream;
        fstream.open("Test_file" + std::to_string(jj), std::ifstream::in | std::ifstream::binary);
        fstream.seekg(0, std::ios::end);
        int filSize = fstream.tellg();
        fstream.close();
        ASSERT_EQ(filSize, 15637);
    }

    }
    dada_buffer.destroy();
}

TEST_F(TestFileWriterTest, test_number_of_files)
/* Test whether the correct number of files are written for a given data size */
{
 /* Setup  the Dada buffers and write to them */
    DadaDB dada_buffer(8, 10240, 4, 4096);
    dada_buffer.create();

    {
    //Setup write clients
    MultiLog log("Output buffer");
    DadaOutputStream outstream(dada_buffer.key(), log);

    // Sigproc Header
    char* hdr_ptr = new char[4096];
    std::memset(hdr_ptr, 0, 4096);
    std::ifstream file("header.txt");
    std::string str((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    const char* hdrstr = str.c_str();
    std::memcpy(hdr_ptr, hdrstr,277);
    RawBytes header(hdr_ptr, 4096, 4096, false);

    file.close();
    // Setup RawBytes to write
    char* data_ptr = new char[10240];
    RawBytes data(data_ptr, 10240, 10240, false);
    // Write data to the buffer
    outstream.init(header);
    for (std::uint32_t ii =0 ; ii < 8; ++ii)
    {
        outstream(data);
    }


    // Write data to file
    MultiLog log1("input stream");
    std::string filename("Fil_file");
    TestFileWriter testfiles(filename,20480);

    DadaReadClient client(dada_buffer.key(), log1);
    auto& header_stream = client.header_stream();
    auto& header_block = header_stream.next();
    testfiles.init(header_block);
    header_stream.release();

    // Read Client
    for (std::uint8_t jj = 0; jj < 8 ; ++jj )
    {
        auto& data_stream = client.data_stream();
        if (data_stream.at_end())
        {
            BOOST_LOG_TRIVIAL(info) << "Reached end of data";
            break;
        }
        auto& data_block = data_stream.next();
        testfiles(data_block);
        data_stream.release();
    }


    /* Check number of files created */
    glob_t glob_result;
    std::memset(&glob_result, 0, sizeof(glob_result));
    int return_value = glob("*Fil_file*", GLOB_TILDE, NULL, &glob_result);
    if(return_value != 0) {
        globfree(&glob_result);
        std::stringstream ss;
        ss << "glob() failed with return_value " << return_value << std::endl;
    }
    std::vector<std::string> filenames;
    for(size_t i = 0; i < glob_result.gl_pathc; ++i) {
        filenames.push_back(std::string(glob_result.gl_pathv[i]));
    }

    globfree(&glob_result);

    ASSERT_EQ(filenames.size(), std::size_t(4));

    }
    dada_buffer.destroy();
}

TEST_F(TestFileWriterTest, test_exception)
/* Test whether the files that are written are of the correct size */
{
/* Setup  the Dada buffers and write to them */
    DadaDB dada_buffer(8, 10240, 4, 4096);
    dada_buffer.create();
    {

    //Setup write clients
    MultiLog log("Output buffer");
    DadaOutputStream outstream(dada_buffer.key(), log);

    // Setup RawBytes to write
    char* hdr_ptr = new char[4096];
    std::memset(hdr_ptr, 0, 4096);
    RawBytes header(hdr_ptr, 4096, 4096, false);
    char* data_ptr = new char[10240];
    RawBytes data(data_ptr, 10240, 10240, false);
    // Write data to the buffer
    outstream.init(header);
    for (std::uint32_t ii =0 ; ii < 8; ++ii)
    {
        outstream(data);
    }


    // Write data to file
    MultiLog log1("input stream");
    std::string filename("Test_file");
    TestFileWriter testfiles(filename,15360);

    DadaReadClient client(dada_buffer.key(), log1);
    auto& header_stream = client.header_stream();
    auto& header_block = header_stream.next();
    ASSERT_ANY_THROW(testfiles.init(header_block));
    header_stream.release();

    }
    dada_buffer.destroy();

}



} //namespace test
} //namespace tuse
} //namespace meerkat
} //namespace psrdada_cpp

