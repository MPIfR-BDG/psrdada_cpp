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

    float centre_frequency() const;
    void centre_frequency(float cfreq);

    float bandwidth() const;
    void bandwidth(float bw);

    std::vector<float> const& channel_frequencies() const;

    // These are all just wrappers to provide programmatic access
    // to the compile time constants that we are forced to use for
    // beamformer perofmance.
    std::size_t cb_tscrunch() const {return FBFUSE_CB_TSCRUNCH;}
    std::size_t cb_fscrunch() const {return FBFUSE_CB_FSCRUNCH;}
    std::size_t cb_nantennas() const {return FBFUSE_CB_NANTENNAS;}
    std::size_t cb_antenna_offset() const {return FBFUSE_CB_ANTENNA_OFFSET;}
    std::size_t cb_nbeams() const {return FBFUSE_CB_NBEAMS;}
    std::size_t cb_nsamples_per_block() const {return FBFUSE_CB_NSAMPLES_PER_BLOCK;}

    /**
     * Below are methods to get and set the power scaling and offset in the beamformer
     * these are tricky parameters that need to be set correctly for the beamformer to
     * function as expected. The values are used when downcasting from floating point
     * to 8-bit integer at the end stage of beamforming. The scaling is the last step
     * in the code and as such the factors can be quite large.
     *
     * The scaling and offset are applied such that:
     *
     *    int8_t scaled_power = static_cast<int8_t>((power - offset) / scaling);
     *
     * In the case above, the variable power is the power after summing all antennas,
     * timesamples and frequency channels requested (tscrunch and fscrunch, respectively).
     * The offset and scaling can be estimated if the power per input F-engine stream is known.
     *
     *    offset = fscrunch * tscrunch * 2 *(input.real.std() * 127 * sqrt(nantennas * npol))**2
     *
     * (the factor 127 comes from the amplitude of the weights which we scale to 127 to allow for
     *  greatest range of possible phase angles from an 16-bit complex number).
     *
     * As the data is chi2 distributed after power generation we can estimate the standard deviation
     * with:
     *
     *    scaling = offset / sqrt(tscrunch * fscrunch * npol)
     *
     * This information would probably be best be encoded with only the standard deviation on the real
     * and imaginary components of the input voltates (on a per channel basis).
     *
     * Note: We do not assume different scaling per channel, if there are significantly different power
     * levels in each channel the scaling should always be set to accommodate the worst cast scenario.
     */
    void output_level(float level);
    float output_level() const;
    void input_level(float level);
    float cb_power_scaling() const;
    float cb_power_offset() const;
    float ib_power_scaling() const;
    float ib_power_offset() const;

    std::size_t ib_tscrunch() const {return FBFUSE_IB_TSCRUNCH;}
    std::size_t ib_fscrunch() const {return FBFUSE_IB_FSCRUNCH;}
    std::size_t ib_nantennas() const {return FBFUSE_IB_NANTENNAS;}
    std::size_t ib_anntena_offset() const {return FBFUSE_IB_ANTENNA_OFFSET;}
    std::size_t ib_nbeams() const {return FBFUSE_IB_NBEAMS;}
    std::size_t total_nantennas() const {return FBFUSE_TOTAL_NANTENNAS;}
    std::size_t nchans() const {return FBFUSE_NCHANS;}
    std::size_t total_nchans() const {return FBFUSE_NCHANS_TOTAL;}
    std::size_t npol() const {return FBFUSE_NPOL;}
    std::size_t nsamples_per_heap() const {return FBFUSE_NSAMPLES_PER_HEAP;}

    void calculate_channel_frequencies();

private:
    std::string _delay_buffer_shm;
    std::string _delay_buffer_mutex;
    std::string _delay_buffer_sem;
    key_t _input_dada_key;
    key_t _cb_dada_key;
    key_t _ib_dada_key;
    float _cfreq;
    float _bw;
    std::vector<float> _channel_frequencies;
    bool _channel_frequencies_stale;
    float _input_level;
    float _output_level;
    float _cb_power_scaling;
    float _cb_power_offset;
    float _ib_power_scaling;
    float _ib_power_offset;
};

} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

#endif //PSRDADA_CPP_MEERKAT_PIPELINECONFIG_HPP
