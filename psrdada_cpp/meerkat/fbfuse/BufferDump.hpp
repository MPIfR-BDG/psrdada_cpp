#ifndef PSRDADA_CPP_MEERKAT_BUFFERDUMP_HPP
#define PSRDADA_CPP_MEERKAT_BUFFERDUMP_HPP

#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/dada_read_client.hpp"
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

template <typename Handler>
class BufferDump
{
    public:
        BufferDump(
            DadaReadClient& client,
            Handler& handler,
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
        void capture(const Event&);
        void read_dada_header();
        void skip_block();
        bool has_event() const;
        void get_event(Event& event);

    private:
        DadaReadClient& _client;
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
        std::unique_ptr<local::stream_protocol::socket> _socket;
};

} // fbfuse
} // meerkat
} // psrdada_cpp

#include "psrdada_cpp/meerkat/fbfuse/detail/BufferDump.cpp"

#endif //PSRDADA_CPP_MEERKAT_BUFFERDUMP_HPP
