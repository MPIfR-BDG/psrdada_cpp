#ifndef PSRDADA_CPP_MEERKAT_PIPELINECONFIG_HPP
#define PSRDADA_CPP_MEERKAT_PIPELINECONFIG_HPP

#include "psrdada/meerkat/fbfuse/PipelineConfig.hpp"
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

} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

#endif //PSRDADA_CPP_MEERKAT_PIPELINECONFIG_HPP