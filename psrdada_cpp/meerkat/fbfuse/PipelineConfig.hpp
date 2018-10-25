#ifndef PSRDADA_CPP_MEERKAT_PIPELINECONFIG_HPP
#define PSRDADA_CPP_MEERKAT_PIPELINECONFIG_HPP

#include "psrdada_cpp/meerkat/fbfuse/fbfuse_constants.hpp"
#include "psrdada_cpp/common.hpp"

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {

class PipelineConfig
{
public:
    PipelineConfig();
    ~PipelineConfig();
    PipelineConfig(PipelineConfig const&) = delete;

    std::string const& delay_buffer_shm() const;
    void delay_buffer_shm(std::string const&);

    std::string const& delay_buffer_mutex() const;
    void delay_buffer_mutex(std::string const&);

    std::string const& delay_buffer_sem() const;
    void delay_buffer_sem(std::string const&);

    key_t input_dada_key() const;
    void input_dada_key(key_t);

    key_t cb_dada_key() const;
    void cb_dada_key(key_t);

    key_t ib_dada_key() const;
    void ib_dada_key(key_t);

    // These are all just wrappers to provide programmatic access
    // to the compile time constants that we are forced to use for
    // beamformer perofmance.
    std::size_t cb_tscrunch() const {return FBFUSE_CB_TSCRUNCH;}
    std::size_t cb_fscrunch() const {return FBFUSE_CB_FSCRUNCH;}
    std::size_t cb_nantennas() const {return FBFUSE_CB_NANTENNAS;}
    std::size_t cb_anntena_offset() const {return FBFUSE_CB_ANTENNA_OFFSET;}
    std::size_t cb_nbeams() const {return FBFUSE_CB_NBEAMS;}
    std::size_t ib_tscrunch() const {return FBFUSE_IB_TSCRUNCH;}
    std::size_t ib_fscrunch() const {return FBFUSE_IB_FSCRUNCH;}
    std::size_t ib_nantennas() const {return FBFUSE_IB_NANTENNAS;}
    std::size_t ib_anntena_offset() const {return FBFUSE_IB_ANTENNA_OFFSET;}
    std::size_t ib_nbeams() const {return FBFUSE_IB_NBEAMS;}
    std::size_t total_nantennas() const {return FBFUSE_TOTAL_NANTENNAS;}
    std::size_t nchans() const {return FBFUSE_NCHANS;}
    std::size_t total_nchans() const {return FBFUSE_NCHANS_TOTAL;}

private:
    std::string _delay_buffer_shm;
    std::string _delay_buffer_mutex;
    std::string _delay_buffer_sem;
    key_t _input_dada_key;
    key_t _cb_dada_key;
    key_t _ib_dada_key;
    std::vector<float> _channel_frequencies;


};

} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

#endif //PSRDADA_CPP_MEERKAT_PIPELINECONFIG_HPP
