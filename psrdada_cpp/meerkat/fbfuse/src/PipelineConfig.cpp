#include "psrdada_cpp/meerkat/fbfuse/PipelineConfig.hpp"
#include <fstream>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {

PipelineConfig::PipelineConfig()
    : _delay_buffer_shm("fbfuse_delays_shm")
    , _delay_buffer_mutex("fbfuse_delays_mutex")
    , _delay_buffer_sem("fbfuse_buffer_counter")
    , _input_dada_key(0xdada)
    , _cb_dada_key(0xcaca)
    , _ib_dada_key(0xeaea)
    , _channel_frequencies_stale(true)
{
}

PipelineConfig::~PipelineConfig()
{
}

std::string const& PipelineConfig::delay_buffer_shm() const
{
    return _delay_buffer_shm;
}

void PipelineConfig::delay_buffer_shm(std::string const& key)
{
    _delay_buffer_shm = key;
}


std::string const& PipelineConfig::delay_buffer_mutex() const
{
    return _delay_buffer_mutex;
}

void PipelineConfig::delay_buffer_mutex(std::string const& key)
{
    _delay_buffer_mutex = key;
}

std::string const& PipelineConfig::delay_buffer_sem() const
{
    return _delay_buffer_sem;
}

void PipelineConfig::delay_buffer_sem(std::string const& key)
{
    _delay_buffer_sem = key;
}

key_t PipelineConfig::input_dada_key() const
{
    return _input_dada_key;
}

void PipelineConfig::input_dada_key(key_t key)
{
    _input_dada_key = key;
}

key_t PipelineConfig::cb_dada_key() const
{
    return _cb_dada_key;
}

void PipelineConfig::cb_dada_key(key_t key)
{
    _cb_dada_key = key;
}

key_t PipelineConfig::ib_dada_key() const
{
    return _ib_dada_key;
}

void PipelineConfig::ib_dada_key(key_t key)
{
    _ib_dada_key = key;
}

float PipelineConfig::centre_frequency() const
{
    return _cfreq;
}

void PipelineConfig::centre_frequency(float cfreq)
{
    _cfreq = cfreq;
    _channel_frequencies_stale = true;
}

float PipelineConfig::bandwidth() const
{
    return _bw;
}

void PipelineConfig::bandwidth(float bw)
{
    _bw = bw;
    _channel_frequencies_stale = true;
}

std::vector<float> const& PipelineConfig::channel_frequencies() const
{
    if (_channel_frequencies_stale)
    {
        throw std::runtime_error("Channel frequencies are stale, "
            "calculate_channel_frequencies() must be called.");
    }
    return _channel_frequencies;
}

void PipelineConfig::calculate_channel_frequencies()
{
    /**
     * Need to revisit this implementation as it is not clear how the
     * frequencies are labeled for the data out of the F-engine. Either
     * way is a roughly correct place-holder.
     */
    float chbw = bandwidth()/nchans();
    float fbottom = centre_frequency() - bandwidth()/2.0f;
    _channel_frequencies.clear();
    for (std::size_t chan_idx=0; chan_idx < nchans(); ++chan_idx)
    {
        _channel_frequencies.push_back(fbottom + chbw/2.0f + (chbw * chan_idx));
    }
    _channel_frequencies_stale = false;
}

} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

