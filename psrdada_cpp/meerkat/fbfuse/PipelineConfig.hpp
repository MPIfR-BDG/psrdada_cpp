#ifndef PSRDADA_CPP_MEERKAT_PIPELINECONFIG_HPP
#define PSRDADA_CPP_MEERKAT_PIPELINECONFIG_HPP

#include "psrdada_cpp/meerkat/fbfuse/fbfuse_constants.hpp"
#include "psrdada_cpp/common.hpp"

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {

/**
 * @brief      Class for wrapping the FBFUSE pipeline configuration.
 */
class PipelineConfig
{
public:
    PipelineConfig();
    ~PipelineConfig();
    PipelineConfig(PipelineConfig const&) = delete;

    /**
     * @brief      Get the key to POSIX shared memory
     *             buffer for delays.
     */
    std::string const& delay_buffer_shm() const;

    /**
     * @brief      Set the key to POSIX shared memory
     *             buffer for delays.
     */
    void delay_buffer_shm(std::string const&);

    /**
     * @brief      Get the key to POSIX mutex
     *             for the delay buffer.
     *
     * @detail     This mutex is used to prevent clients
     *             from reading the delay buffer during
     *             and update.
     */
    std::string const& delay_buffer_mutex() const;

    /**
     * @brief      Set the key to POSIX mutex
     *             for the delay buffer.
     *
     * @detail     This mutex is used to prevent clients
     *             from reading the delay buffer during
     *             and update.
     */
    void delay_buffer_mutex(std::string const&);

    /**
     * @brief      Get the key to POSIX semaphore
     *             for the delay buffer.
     *
     * @detail     This is a counting semaphore that
     *             is updated whenever a new delay
     *             model becomes available.
     */
    std::string const& delay_buffer_sem() const;

    /**
     * @brief      Set the key to POSIX semaphore
     *             for the delay buffer.
     *
     * @detail     This is a counting semaphore that
     *             is updated whenever a new delay
     *             model becomes available.
     */
    void delay_buffer_sem(std::string const&);

    /**
     * @brief      Get the DADA key for the input buffer
     */
    key_t input_dada_key() const;

    /**
     * @brief      Set the DADA key for the input buffer
     */
    void input_dada_key(key_t);

    /**
     * @brief      Get the DADA key for the output buffer
     *             (for coherent beam data)
     */
    key_t cb_dada_key() const;

    /**
     * @brief      Set the DADA key for the output buffer
     *             (for coherent beam data)
     */
    void cb_dada_key(key_t);

    /**
     * @brief      Get the DADA key for the output buffer
     *             (for incoherent beam data)
     */
    key_t ib_dada_key() const;

    /**
     * @brief      Set the DADA key for the output buffer
     *             (for incoherent beam data)
     */
    void ib_dada_key(key_t);

    /**
     * @brief      Get the centre frequency for the subband to
     *             be processed by this instance.
     */
    float centre_frequency() const;

    /**
     * @brief      Set the centre frequency for the subband to
     *             be processed by this instance.
     */
    void centre_frequency(float cfreq);

    /**
     * @brief      Get the bandwidth of the subband to
     *             be processed by this instance.
     */
    float bandwidth() const;

    /**
     * @brief      Set the bandwidth of the subband to
     *             be processed by this instance.
     */
    void bandwidth(float bw);

    /**
     * @brief      Return the centre frequency of each channel in the
     *             subband to be processed.
     */
    std::vector<float> const& channel_frequencies() const;

    // These are all just wrappers to provide programmatic access
    // to the compile time constants that we are forced to use for
    // beamformer performance.

    /**
     * @brief      Return the number of time samples to be integrated
     *             in the coherent beamformer.
     */
    std::size_t cb_tscrunch() const {return FBFUSE_CB_TSCRUNCH;}

    /**
     * @brief      Return the number of frequency channels to be integrated
     *             in the coherent beamformer.
     */
    std::size_t cb_fscrunch() const {return FBFUSE_CB_FSCRUNCH;}

    /**
     * @brief      Return the number of antennas to use for the coherent
     *             beamformer.
     */
    std::size_t cb_nantennas() const {return FBFUSE_CB_NANTENNAS;}

    /**
     * @brief      Return the index of the first antenna in the set of antennas
     *             to be used in the coherent beamformer.
     */
    std::size_t cb_antenna_offset() const {return FBFUSE_CB_ANTENNA_OFFSET;}

    /**
     * @brief      Return the number of beams to be formed by the coherent beamformer
     */
    std::size_t cb_nbeams() const {return FBFUSE_CB_NBEAMS;}

    /**
     * @brief      Return the number of samples that will be processed per block in the
     *             coherent beamformer kernels
     *
     * @note       This is very specific and ties the design heavily to the architecture
     *             of the beamforming kernel (this is bad design but not catastrophic).
     */
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
     * The interested reader can examine the PipelineConfig::update_power_offsets_and_scalings
     * method to see how the exact scaling and offsets are calculated for the coherent and incoherent
     * beamformer
     *
     * Note: We do not assume different scaling per channel, if there are significantly different power
     * levels in each channel the scaling should always be set to accommodate the worst cast scenario.
     */

    /**
     * @brief      Set the output standard deviation for data out
     *             of both the coherent and incoherent beamformers
     */
    void output_level(float level);

    /**
     * @brief      Get the output standard deviation for data out
     *             of both the coherent and incoherent beamformers
     */
    float output_level() const;

    /**
     * @brief      Set the input standard deviation for data into
     *             both the coherent and incoherent beamformers.
     *
     * @note       This is the standard deviation on the real/imag
     *             components of the input F-engine data.
     */
    void input_level(float level);

    /**
     * @brief      Get the coherent beamformer power scaling
     */
    float cb_power_scaling() const;

    /**
     * @brief      Get the coherent beamformer power offset
     */
    float cb_power_offset() const;

    /**
     * @brief      Get the incoherent beamformer power scaling
     */
    float ib_power_scaling() const;

    /**
     * @brief      Get the incoherent beamformer power offset
     */
    float ib_power_offset() const;

    /**
     * @brief      Return the number of time samples to be integrated
     *             in the incoherent beamformer.
     */
    std::size_t ib_tscrunch() const {return FBFUSE_IB_TSCRUNCH;}

    /**
     * @brief      Return the number of frequency channels to be integrated
     *             in the incoherent beamformer.
     */
    std::size_t ib_fscrunch() const {return FBFUSE_IB_FSCRUNCH;}

    /**
     * @brief      Return the number of antennas to use for the incoherent
     *             beamformer.
     */
    std::size_t ib_nantennas() const {return FBFUSE_IB_NANTENNAS;}

    /**
     * @brief      Return the index of the first antenna in the set of antennas
     *             to be used in the coherent beamformer.
     *
     * @note       This parameter currently as NO EFFECT as the incoherent beamformer
     *             currently assumes that all antennas are intended to be included
     *             in the incoherent beam.
     */
    std::size_t ib_anntena_offset() const {return FBFUSE_IB_ANTENNA_OFFSET;}

    /**
     * @brief      Return the number of incoherent beams to be produced
     *
     * @note       This parameter currently as NO EFFECT as the incoherent beamformer
     *             currently produces only one beam.
     *             TODO: Delete this parameter.
     */
    std::size_t ib_nbeams() const {return FBFUSE_IB_NBEAMS;}

    /**
     * @brief      Return the total number of antennas in the input data
     */
    std::size_t total_nantennas() const {return FBFUSE_TOTAL_NANTENNAS;}

    /**
     * @brief      Return the total number of frequency channels in the input data
     */
    std::size_t nchans() const {return FBFUSE_NCHANS;}

    /**
     * @brief      Return the total number of frequency channels in the
     *             observation.
     *
     * @note       This corresponds to the mode of the MeerKAT F-engines and is
     *             used to correctly work out the timestamp change between blocks
     *             of received data. This should probably be replaced with the
     *             sampling clock as an input.
     */
    std::size_t total_nchans() const {return FBFUSE_NCHANS_TOTAL;}

    /**
     * @brief      Return the number of polarisations in the observation
     *
     * @note       This better be 2 otherwise who knows what will happen...
     */
    std::size_t npol() const {return FBFUSE_NPOL;}

    /**
     * @brief      Return the number of time samples per F-engine SPEAD heap.
     *
     * @note       This corresponds to the inner "T" dimension in the input
     *             TAF[T]P order data.
     */
    std::size_t nsamples_per_heap() const {return FBFUSE_NSAMPLES_PER_HEAP;}

private:
    void calculate_channel_frequencies() const;
    void update_power_offsets_and_scalings();

private:
    std::string _delay_buffer_shm;
    std::string _delay_buffer_mutex;
    std::string _delay_buffer_sem;
    key_t _input_dada_key;
    key_t _cb_dada_key;
    key_t _ib_dada_key;
    double _cfreq;
    double _bw;
    mutable std::vector<double> _channel_frequencies;
    mutable bool _channel_frequencies_stale;
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
