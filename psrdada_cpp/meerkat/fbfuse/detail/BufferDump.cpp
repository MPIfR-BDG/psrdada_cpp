#include "psrdada_cpp/meerkat/fbfuse/BufferDump.hpp"
#include "psrdada_cpp/meerkat/fbfuse/Header.hpp"
#include "psrdada_cpp/raw_bytes.hpp"
#include<cstdio>
#include<cstdlib>

using namespace std;
//using boost::property_tree::ptree;

/* A quick C++ script that calculates the DM delay and snips out the data accordingly to store
 */

namespace
{
    double dm_delay( float f1, float f2, float dm, float tsamp)
    {
        return (4.15 * 0.001 * dm * (std::powl(1/f1,2) - std::powl(1/f2, 2)))/tsamp;
    }
}

template <typename Handler>
BufferDump<Handler>::BufferDump(
    DadaReadClient& client,
    Handler& handler,
    float max_fill_level,
    std::size_t nantennas,
    std::size_t subband_nchannels,
    std::size_t total_nchannels,
    float centre_freq,
    float bandwidth)
    : _client(client)
    , _handler(handler)
    , _max_fill_level(max_fill_level)
    , _nantennas(nantennas)
    , _subband_nchans(subband_nchannels)
    , _total_nchans(total_nchannels)
    , _centre_freq(centre_freq)
    , _bw(bandwidth)
    , _current_block_idx(0)
    , _stop(false)
{
    std::memset(_event_msg_buffer, 0, sizeof(_event_msg_buffer));
    std::memset(_header_buffer, 0, sizeof(_header_buffer));
}

template <typename Handler>
BufferDump<Handler>::~BufferDump()
{
}

template <typename Handler>
void BufferDump<Handler>::start()
{
    BOOST_LOG_TRIVIAL(info) << "Starting BufferDump instance (listenting on socket '')";
    // Open Unix socket endpoint
    /**
    ::unlink("/tmp/foobar"); // Remove previous binding.
    local::stream_protocol::endpoint ep("/tmp/foobar");
    local::stream_protocol::acceptor acceptor(my_io_context, ep);
    local::stream_protocol::socket socket(my_io_context);
    acceptor.accept(socket);
    */
    read_dada_header();
    listen();
}

template <typename Handler>
void BufferDump<Handler>::stop()
{
    _stop = true;
}

template <typename Handler>
void BufferDump<Handler>::read_dada_header();
{
    BOOST_LOG_TRIVIAL(info) << "Parsing DADA buffer header";
    auto& header_block = _client.header_steam().next();
    Header parser(header_block);
    _sample_clock_start = parser.get<std::size_t>("SAMPLE_CLOCK_START");
    _sample_clock = parser.get<std::size_t>("SAMPLE_CLOCK");
    _sync_time = parser.get<long double>("SYNC_TIME");
    BOOST_LOG_TRIVIAL(info) << "Parsed SAMPLE_CLOCK_START = " << _sample_clock_start;
    BOOST_LOG_TRIVIAL(info) << "Parsed SAMPLE_CLOCK = " << _sample_clock;
    BOOST_LOG_TRIVIAL(info) << "Parsed SYNC_TIME = " << _sync_time;
    std::memcpy(_header_buffer, header.ptr(), sizeof(_header_buffer));
    _client.header_steam().release();
    BOOST_LOG_TRIVIAL(info) << "Parsing complete";
}

template <typename Handler>
void BufferDump<Handler>::listen()
{
    while (!_stop)
    {
        if has_event()
        {
            Event event;
            try
            {
                get_event(event);
            }
            catch(std::exception& e)
            {
                BOOST_LOG_TRIVIAL(error) << e.what();
                continue;
            }
            capture(event);
        }

        float fill_level = _client.data_buffer_percent_full();
        BOOST_LOG_TRIVIAL(debug) << "DADA buffer fill level = " << fill_level << "%";
        while (fill_level > _max_fill_level)
        {
            skip_block();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

template <typename Handler>
void BufferDump<Handler>::skip_block()
{
    BOOST_LOG_TRIVIAL(debug) << "Skipping DADA block";
    _client.data_stream().next();
    _client.data_stream().release();
    _current_block_idx += 1;
    BOOST_LOG_TRIVIAL(debug) << "Current block IDX = " << _current_block_idx;
}

template <typename Handler>
bool BufferDump<Handler>::has_event() const
{
    return false;
}

template <typename Handler>
void BufferDump<Handler>::get_event(Event& event)
{
    /**
    ptree pt;
    boost::asio::read(_socket, boost::asio::buffer(
        _event_msg_buffer, sizeof(_event_msg_buffer)));
    std::string event_string(_event_msg_buffer);
    boost::property_tree::json_parser::read_json(event_string, pt);
    event.utc_start = pt.get<long double>("utc_start");
    event.utc_end = pt.get<long double>("utc_end");
    event.dm = pt.get<float>("dm");
    event.reference_freq = pt.get<float>("reference_freq");
    event.trigger_id = pt.get<std::string>("trigger_id");
    */

}

template <typename Handler>
void BufferDump<Handler>::capture(Event const& event)
{

    std::size_t output_left_idx = 0, output_right_idx = 0;
    std::size_t block_left_idx = 0, block_right_idx = 0;
    std::vector<std::size_t> left_edge_of_output(_subband_nchans);
    std::vector<std::size_t> right_edge_of_output(_subband_nchans);
    long double start_of_buffer = _sync_time + _sample_clock_start / _sample_clock;
    BOOST_LOG_TRIVIAL(debug) << "Unix time at start of DADA buffer = " << start_of_buffer;
    long double tsamp = (2 * _total_nchans) / _sample_clock;
    BOOST_LOG_TRIVIAL(debug) << "Sample interval = " << tsamp;
    std::size_t start_sample = static_cast<std::size_t>((event.utc_start - start_of_buffer) / tsamp);
    BOOST_LOG_TRIVIAL(debug) << "First input sample in output block (@reference freq) = " << start_sample;
    std::size_t end_sample = static_cast<std::size_t>((event.utc_end - start_of_buffer) / tsamp);
    BOOST_LOG_TRIVIAL(debug) << "Last input sample in output block (@reference freq) = " << end_sample;
    std::size_t nsamps = end_sample - start_sample;
    BOOST_LOG_TRIVIAL(debug) << "Number of timesamples in output = " << nsamps;
    float chan_bw = _bw / _subband_nchans;
    BOOST_LOG_TRIVIAL(debug) << "Channel bandwidth = " << chan_bw;
    std::size_t nelements = _subband_nchans * _nantennas * nsamps;
    BOOST_LOG_TRIVIAL(debug) << "Resizing output buffer to " << nelements << " elements";
    _tmp_buffer.resize(nelements);

    BOOST_LOG_TRIVIAL(debug) << "Calculating sample offsets for each channel";
    for (std::size_t ii = 0; ii < _subband_nchans; ++ii)
    {
        float channel_freq = (_centre_freq - chan_bw/2.0f) + ii * chan_bw;
        double delay = dm_delay(event.reference_freq, channel_freq, event.dm, _tsamp);
        left_edge_of_output[ii] = static_cast<std::size_t>(delay) + start_sample;
        right_edge_of_output[ii] = static_cast<std::size_t>(delay) + end_sample;
        BOOST_LOG_TRIVIAL(debug) << "Channel " << ii << ", "
                                 << "delay = " << delay << ", "
                                 << "span = (" << left_edge_of_output[ii] << ", "
                                 << right_edge_of_output[ii] << ")";
    }

    std::size_t start_block_idx = left_edge_of_output[0] / _samples_per_block;
    std::size_t end_block_idx = right_edge_of_output[_subband_nchans-1] / _samples_per_block;
    BOOST_LOG_TRIVIAL(debug) << "First DADA block to extract from = " << start_block_idx;
    BOOST_LOG_TRIVIAL(debug) << "Last DADA block to extract from = " << end_block_idx;
    while (_current_block_idx < start_block_idx)
    {
        skip_block();
    }
    std::size_t i_t = 256;
    std::size_t i_ft = _subband_nchans * i_t;
    std::size_t i_aft = _nantennas * i_ft;
    std::size_t o_t = nsamps;
    std::size_t o_at = _nantennas * o_t;

    while (_current_block_idx <= end_block_idx)
    {
        BOOST_LOG_TRIVIAL(debug) << "Extracting data from block " << _current_block_idx;
        RawBytes& block = _client.data_stream().next();
        std::size_t block_start = _current_block_idx * _samples_per_block;
        std::size_t block_end = (_current_block_idx + 1) * _samples_per_block;
        for (std::size_t chan_idx = 0; chan_idx < _subband_nchans; ++chan_idx)
        {
            if ((left_edge_of_output[chan_idx] > block_end) ||
                (right_edge_of_output[chan_idx] < block_start))
            {
                continue;
            }

            if (block_start <= left_edge_of_output[chan_idx])
            {
                output_left_idx = 0;
                block_left_idx = left_edge_of_output[chan_idx] - block_start;
            }
            else
            {
                output_left_idx = block_start - left_edge_of_output[chan_idx];
                block_left_idx = 0;
            }

            if (block_end >= right_edge_of_output[chan_idx])
            {
                output_right_idx = nsamps;
                block_right_idx = right_edge_of_output[chan_idx] - block_start;
            }
            else
            {
                output_right_idx = block_end - left_edge_of_output[chan_idx];
                block_right_idx = _samples_per_block;
            }

            std::size_t span = output_right_idx - output_left_idx;
            for (std::size_t span_idx = 0; span_idx < span; ++span_idx)
            {
                std::size_t output_t = output_left_idx + span_idx;
                std::size_t input_t = block_left_idx + span_idx;
                std::size_t group_t = input_t / i_t;
                std::size_t offset_t = input_t % i_t;
                std::size_t input_idx = group_t * i_aft + chan_idx * i_t + offset_t;
                std::size_t output_idx = chan_idx * o_at + output_t;
                for (std::size_t antenna_idx = 0; antenna_idx < _nantennas; ++antenna_idx)
                {
                    _tmp_buffer[output_idx + antenna_idx * o_t] = block.ptr()[input_idx + antenna_idx * i_ft];
                }
            }
        }
        _client.data_stream().release();
        _current_block_idx += 1;
    }
    BOOST_LOG_TRIVIAL(debug) << "Updating header";
    RawBytes header(_header_buffer, 4096, 4096);
    Header parser(header);
    std::size_t sample_clock_start = start_sample * _total_nchans * 2;
    parser.set<std::size_t>("SAMPLE_CLOCK_START", sample_clock_start);
    _handler.init(header);
    BOOST_LOG_TRIVIAL(debug) << "Outputing data to handler";
    std::size_t nbytes = _tmp_buffer.size() * sizeof(char4);
    RawBytes output((char*) _tmp_buffer.data(), nbytes, nbytes);
    _handler(output);
}




