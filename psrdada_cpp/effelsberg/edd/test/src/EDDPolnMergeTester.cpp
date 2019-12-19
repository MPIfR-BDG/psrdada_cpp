#include "psrdada_cpp/effelsberg/edd/test/EDDPolnMergeTester.hpp"
#include "psrdada_cpp/effelsberg/edd/EDDPolnMerge.hpp"
#include "psrdada_cpp/dada_write_client.hpp"
#include "psrdada_cpp/dada_read_client.hpp"
#include "psrdada_cpp/dada_db.hpp"
#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/multilog.hpp"
#include "ascii_header.h"
#include <vector>

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {
namespace test {

EDDPolnMergeTester::EDDPolnMergeTester()
    : ::testing::Test()
{
}

EDDPolnMergeTester::~EDDPolnMergeTester()
{
}

void EDDPolnMergeTester::SetUp()
{
}

void EDDPolnMergeTester::TearDown()
{
}

TEST_F(EDDPolnMergeTester, test_sequence)
{
	std::size_t ii, jj, kk;
	std::size_t npol = 2;
	std::size_t nsamps_per_heaps = 4096;
	std::size_t nheap_groups = 32;
	std::size_t buffer_size = npol * nsamps_per_heaps * nheap_groups;
	std::vector<char> test_vector(buffer_size);
	DadaDB db(4, buffer_size, 8, 4096);
	db.create();
        MultiLog log("edd::EDDPolnMerge_test");
	DadaReadClient reader(db.key(), log);
	DadaWriteClient writer(db.key(), log);

	
	for (ii = 0; ii < nheap_groups; ii++)
	{
		for (jj = 0; jj < npol; jj++)
		{
			for (kk = 0; kk < nsamps_per_heaps; kk++)
			{
				test_vector[ii * npol * nsamps_per_heaps + jj * nsamps_per_heaps + kk]  = (ii * 2 + jj) % 128;
			}
		}	
	}
	RawBytes input(test_vector.data(), buffer_size, buffer_size);
	effelsberg::edd::EDDPolnMerge merger(nsamps_per_heaps, npol, writer);
	std::vector<char> test_header(4096);
	RawBytes input_header(test_header.data(), 4096, 4096); 
	ascii_header_set(input_header.ptr(), "CLOCK_SAMPLE", "%s", "3200000000");
	ascii_header_set(input_header.ptr(), "SAMPLE_CLOCK_START", "%s", "1000000000");
//	BOOST_LOG_TRIVIAL(debug) <<input_header;
	//ascii_header_set(input_header.ptr(), "SAMPLE_CLOCKt", "%s", "2400000000");
	//ascii_header_set(input_header.ptr(), "SAMPLE_CLOCK_TEST", "%s", "3200000000");
        //ascii_header_set(input_header.ptr(), "SAMPLE_CLOCK_START", "%s", "1000000000");
        //ascii_header_set(input_header.ptr(), "SAMPLE_CLOCK_START_BLAH", "%s", "tesing");
	ascii_header_set(input_header.ptr(), "SYNC_TIME", "%s", "1574694522.0");
	char buffer[1024];
	ascii_header_get(input_header.ptr(), "SYNC_TIME ", "%s", buffer);
	BOOST_LOG_TRIVIAL(debug) << "SYNC_TIME " << buffer;
	//ascii_header_get(input_header.ptr(), "SAMPLE_CLOCKt", "%s", buffer);
	//BOOST_LOG_TRIVIAL(debug) << "SAMPLE_CLOCKt" << buffer;
        ascii_header_get(input_header.ptr(), "CLOCK_SAMPLE", "%s", buffer);
        BOOST_LOG_TRIVIAL(debug) << "CLOCK_SAMPLE " << buffer;
        ascii_header_get(input_header.ptr(), "SAMPLE_CLOCK_START", "%s", buffer);
	BOOST_LOG_TRIVIAL(debug) << "SAMPLE_CLOCK_START " << buffer;
        //ascii_header_get(input_header.ptr(), "SAMPLE_CLOCK_START_BLAH", "%s", buffer);
        //BOOST_LOG_TRIVIAL(debug) << "SAMPLE_CLOCK_START_BLAH" << buffer;
	
	merger.init(input_header);
	for (std::size_t yy = 0; yy < 2; yy++)
	{
		merger(input);
		RawBytes& block = reader.data_stream().next();
        	for (ii = 0; ii < nheap_groups; ii++)
        	{	
                	for (jj = 0; jj < nsamps_per_heaps; jj++)
                	{
                        	for (kk = 0; kk < npol; kk++)
                        	{
                                	ASSERT_EQ(block.ptr()[ii * npol * nsamps_per_heaps + jj * npol + kk], (ii * 2 + kk) % 128);
                        	}
                	}
        	}
		reader.data_stream().release();
	}
}

} //namespace test
} //namespace edd
} //namespace effelsberg
} //namespace psrdada_cpp

