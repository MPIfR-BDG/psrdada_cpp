#include "psrdada_cpp/effelsberg/edd/EDDPolnMerge.hpp"
#include "ascii_header.h"
#include <immintrin.h>
#include <time.h>
#include <iomanip>
#include <cmath>

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {

	uint64_t interleave(uint32_t x, uint32_t y) {
  	__m128i xvec = _mm_cvtsi32_si128(x);
	__m128i yvec = _mm_cvtsi32_si128(y);
 	__m128i interleaved = _mm_unpacklo_epi8(yvec, xvec);
	  return _mm_cvtsi128_si64(interleaved);
	}

    EDDPolnMerge::EDDPolnMerge(std::size_t nsamps_per_heap, std::size_t npol, DadaWriteClient& writer)
    : _nsamps_per_heap(nsamps_per_heap)
    , _npol(npol)
    , _writer(writer)
    {
    }

    EDDPolnMerge::~EDDPolnMerge()
    {
    }

    void EDDPolnMerge::init(RawBytes& block)
    {
        RawBytes& oblock = _writer.header_stream().next();
        if (block.used_bytes() > oblock.total_bytes())
	{
	    _writer.header_stream().release();
	    throw std::runtime_error("Output DADA buffer does not have enough space for header");
	}
        std::memcpy(oblock.ptr(), block.ptr(), block.used_bytes());
        char buffer[1024];
	ascii_header_get(block.ptr(), "SAMPLE_CLOCK_START", "%s", buffer);
	std::size_t sample_clock_start = std::strtoul(buffer, NULL, 0);
	ascii_header_get(block.ptr(), "CLOCK_SAMPLE", "%s", buffer);
	long double sample_clock = std::strtold(buffer, NULL);
	ascii_header_get(block.ptr(), "SYNC_TIME", "%s", buffer);
	long double sync_time = std::strtold(buffer, NULL);
        BOOST_LOG_TRIVIAL(debug) << "this is sample_clock_start "<< sample_clock_start;
        BOOST_LOG_TRIVIAL(debug)<< "this is sample_clock "<< sample_clock;
        BOOST_LOG_TRIVIAL(debug) << "this is sync_time "<< sync_time;
	BOOST_LOG_TRIVIAL(debug) << "this is sample_clock_start / sample_clock "<< sample_clock_start / sample_clock;
	long double unix_time = sync_time + (sample_clock_start / sample_clock);
	long double mjd_time = unix_time / 86400 - 40587.5;
	char time_buffer[80];
	std::time_t unix_time_int;
	struct std::tm * timeinfo;
	double fractpart, intpart;
	fractpart = std::modf (static_cast<double>(unix_time) , &intpart);
	unix_time_int = static_cast<std::time_t>(intpart);
	timeinfo = std::gmtime (&unix_time_int);
	std::strftime(time_buffer, 80, "%Y-%m-%d-%H:%M:%S", timeinfo);
	std::stringstream utc_time_stamp;
	BOOST_LOG_TRIVIAL(debug) << "unix_time" << unix_time;
	BOOST_LOG_TRIVIAL(debug) << "fractional part " << fractpart;
        //BOOST_LOG_TRIVIAL(debug) << "fractional part ." << static_cast<std::size_t>(fractpart*10000000000);
        //utc_time_stamp<< time_buffer << "." <<fractpart;
	utc_time_stamp<< time_buffer << "." << std::setw(10) << std::setfill('0') << std::size_t(fractpart*10000000000) << std::setfill(' ');
	//BOOST_LOG_TRIVIAL(debug) << "fractional part" <<static_cast<std::size_t>(fractpart * 10000000000);
	//utc_time_stamp<< time_buffer << "." << static_cast<std::size_t>(fractpart * 10000000000);
	BOOST_LOG_TRIVIAL(debug) << "this is start time in utc "<< utc_time_stamp.str().c_str()<< "\n";
//	std::cout << "this is sync_time MJD "<< mjd_time<< "\n";
	ascii_header_set(oblock.ptr(), "UTC_START", "%s", utc_time_stamp.str().c_str());
	ascii_header_set(oblock.ptr(), "UNIX_TIME", "%Lf", unix_time);
	oblock.used_bytes(oblock.total_bytes());
        _writer.header_stream().release();
    }

    bool EDDPolnMerge::operator()(RawBytes& block)
    {
	std:size_t nheap_groups = block.used_bytes()/_npol/_nsamps_per_heap;
/**
        if (block.used_bytes() < block.total_bytes())
        {
            BOOST_LOG_TRIVIAL (debug) << "Reach end of data";
            _writer.data_stream().next();
            _writer.data_stream().release();
            return true;
        }
**/
	RawBytes& oblock = _writer.data_stream().next();

        if (block.used_bytes() > oblock.total_bytes())
        {
	    _writer.data_stream().release();
	    throw std::runtime_error("Output DADA buffer does not match with the input dada buffer");
        }

	uint32_t* S0 = reinterpret_cast<uint32_t*>(block.ptr());
        uint32_t* S1 = reinterpret_cast<uint32_t*>(block.ptr() + _nsamps_per_heap);
	uint64_t* D = reinterpret_cast<uint64_t*>(oblock.ptr());

	for (std::size_t jj = 0; jj < nheap_groups; ++jj)
		{
		for (std::size_t ii = 0; ii < _nsamps_per_heap/sizeof(uint32_t); ++ii)
			{
			*D++ = interleave(*S1++, *S0++);
			}
		S0 += _nsamps_per_heap/sizeof(uint32_t);
		S1 += _nsamps_per_heap/sizeof(uint32_t);
		}

	oblock.used_bytes(block.used_bytes());
	_writer.data_stream().release();
        return false;
    }
}//edd
}//effelsberg
}//psrdada_cpp

