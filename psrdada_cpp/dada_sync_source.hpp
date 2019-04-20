#ifndef PSRDADA_CPP_DADA_SYNC_SOURCE_HPP
#define PSRDADA_CPP_DADA_SYNC_SOURCE_HPP

#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/common.hpp"
#include "ascii_header.h"
#include <vector>
#include <chrono>
#include <string>
#include <iomanip>        // std::put_time
#include <thread>         // std::this_thread::sleep_until
#include <chrono>         // std::chrono::system_clock
#include <ctime>
#include <cmath>

namespace psrdada_cpp
{
    template <class Handler>
    void sync_source(Handler& handler,
        std::size_t header_size,
        std::string const& header_fname,
        std::size_t nbytes_per_write,
        std::size_t total_bytes,
        std::time_t sync_epoch,
        double block_duration,
        std::size_t ts_per_block)
    {
        std::vector<char> sync_header(header_size, 0);
        std::vector<char> sync_block(nbytes_per_write, 1);
        RawBytes header(sync_header.data(), header_size, header_size, false);
        if (header_fname != "")
        {
            std::ifstream headerfile (header_fname);
            if (!headerfile.is_open())
            {
                throw std::runtime_error("Unable to open header file");
            }
            headerfile.read(header.ptr(), header.total_bytes());
            headerfile.close();
        }
        RawBytes data(sync_block.data(), nbytes_per_write, nbytes_per_write, false);
        std::size_t sample_clock_start;
        auto sync_epoch_tp = std::chrono::system_clock::from_time_t(sync_epoch);
        auto curr_epoch_tp =  std::chrono::system_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::microseconds>(curr_epoch_tp - sync_epoch_tp).count();

        std::size_t next_block_idx;

        if (sync_epoch_tp > curr_epoch_tp)
        {
            BOOST_LOG_TRIVIAL(info) << "The sync epoch is " << static_cast<float>(diff)/1e6
            << " seconds in the future";
            next_block_idx = 0;
            sample_clock_start = 0;
        }
        else
        {
            next_block_idx = static_cast<std::size_t>((std::ceil(static_cast<double>(diff)/1e6) / block_duration));
            sample_clock_start = next_block_idx * ts_per_block;
        }

        BOOST_LOG_TRIVIAL(info) << "Setting SAMPLE_CLOCK_START to " << sample_clock_start;
        ascii_header_set(sync_header.data(), "SAMPLE_CLOCK_START", "%ld", sample_clock_start);
        handler.init(header);
        std::size_t bytes_written = 0;

        bool infinite = (total_bytes == 0);
        while (true)
        {
            auto epoch_of_wait = sync_epoch_tp + std::chrono::duration<double>(next_block_idx * block_duration);
            std::this_thread::sleep_until(epoch_of_wait);
            handler(data);
            bytes_written += data.used_bytes();
            ++next_block_idx;
	    if (!infinite && bytes_written >= total_bytes)
	    {
		break;
	    }
        }
    }
} //namespace psrdada_cpp
#endif //PSRDADA_CPP_DADA_SYNC_SOURCE_HPP
