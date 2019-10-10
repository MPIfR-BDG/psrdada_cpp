#ifndef PSRDADA_CPP_MEERKAT_BUFFERDUMP_HPP
#define PSRDADA_CPP_MEERKAT_BUFFERDUMP_HPP

#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/dada_read_client.hpp"
#include <boost/asio.hpp>
#include <string>
#include <vector>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {

    struct Event
    {
        long double utc_start;
        long double utc_end;
        float dm;
        float reference_freq;
        std::string trigger_id;
    };


/**
 * @brief: A class to extract buffer of voltage dumps. It expects a JSON trigger with the essential quantities to extract the data from the DADA buffer.
 *         The JSON metadata expected is as follows
 *         1) start time (utc): UTC start time of the burst at the reference frequency. Note that the start time will change depending on how many samples to save before the burst.
 *         2) end time (utc) :  UTC end time of the burst at the reference frequency. Note that the end time will change depending on how many samples to save after the burst.
 *         3) DM:  Dispersion Measure of the burst.
 *         4) Reference Frequency: Reference Frequency is the highest frequency of the subband in question
 *         5) Trigger ID: UNique ID for the trigger
 **/

/**
 * @brief:  The arguments are:
 *          client: Read Client
 *          handler: Handler to pass extracted data to
 *          max_fill_level: % of buffer to be filled before skipping to the next block
 *          nantennas: Number of antennas
 *          subband_nchannels: Number of frequency channels per subband
 *          total_nchannels: Total number of channels
 *          centre_freq: centre frequency of the current subband
 *          bandwidth: Bandwidth of the subband
 **/

 
    template <typename Handler>
        class BufferDump
        {
            public:
                BufferDump(
                        key_t key,
                        MultiLog& log,
                        Handler& handler,
                        std::string socket_name,
                        float max_fill_level,
                        std::size_t nantennas,
                        std::size_t subband_nchannels,
                        std::size_t total_nchannels,
                        float centre_freq,
                        float bandwidth);
                ~BufferDump();
                void start();
                void stop();

            private:
                void listen();
                void setup();
                void capture(const Event&);
                void read_dada_header();
                void skip_block();
                bool has_event() const;
                void get_event(Event& event);

            private:
                std::unique_ptr<DadaReadClient> _client;
                Handler& _handler;
                std::string _socket_name;
                float _max_fill_level;
                std::size_t _nantennas;
                std::size_t _subband_nchans;
                std::size_t _total_nchans;
                float _centre_freq;
                float _bw;
                std::size_t _current_block_idx;
                bool _stop;
                std::vector<unsigned> _tmp_buffer;
                char _event_msg_buffer[4096];
                char _header_buffer[4096];
                std::size_t _sample_clock_start;
                std::size_t _sample_clock;
                long double _sync_time;
                boost::asio::io_service _io_service;
                std::unique_ptr<boost::asio::local::stream_protocol::acceptor> _acceptor;
                std::unique_ptr<boost::asio::local::stream_protocol::socket> _socket;
        };

} // fbfuse
} // meerkat
} // psrdada_cpp

#include "psrdada_cpp/meerkat/fbfuse/detail/BufferDump.cpp"

#endif //PSRDADA_CPP_MEERKAT_BUFFERDUMP_HPP
