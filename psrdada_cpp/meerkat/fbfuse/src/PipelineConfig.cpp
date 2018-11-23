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
    , _input_level(32.0f)
    , _output_level(24.0f)
    , _cb_power_scaling(0.0f)	
    , _cb_power_offset(0.0f)
    , _ib_power_scaling(0.0f)
    , _ib_power_offset(0.0f)
{
    input_level(_input_level);
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

void PipelineConfig::output_level(float level)
{
    _output_level = level;
    update_power_offsets_and_scalings();
}

float PipelineConfig::output_level() const
{
    return _output_level;
}

void PipelineConfig::input_level(float level)
{
    _input_level = level;
    update_power_offsets_and_scalings();
}

void PipelineConfig::update_power_offsets_and_scalings()
{
    // scalings for coherent beamformer
    const float weights_amp = 127.0f;
    float cb_scale = std::pow(weights_amp * _input_level
        * std::sqrt(static_cast<float>(cb_nantennas())), 2);
    float cb_dof = 2 * cb_tscrunch() * cb_fscrunch() * npol();
    _cb_power_offset = cb_scale * cb_dof;
    _cb_power_scaling = cb_scale * std::sqrt(2 * cb_dof) / _output_level;

    // scaling for incoherent beamformer
    float ib_scale = std::pow(_input_level, 2);
    float ib_dof = 2 * ib_tscrunch() * ib_fscrunch() * ib_nantennas() * npol();
    _ib_power_offset  = ib_scale * ib_dof;
    _ib_power_scaling = ib_scale * std::sqrt(2 * ib_dof) / _output_level;
}

float PipelineConfig::cb_power_scaling() const
{
    return _cb_power_scaling;
}

float PipelineConfig::cb_power_offset() const
{
    return _cb_power_offset;
}

float PipelineConfig::ib_power_scaling() const
{
    return _ib_power_scaling;
}

float PipelineConfig::ib_power_offset() const
{
    return _ib_power_offset;
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

