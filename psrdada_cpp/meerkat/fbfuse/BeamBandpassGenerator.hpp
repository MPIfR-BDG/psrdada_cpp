#ifndef PSRDADA_CPP_MEERKAT_FBFUSE_BEAMBANDPASSGENERATOR_HPP
#define PSRDADA_CPP_MEERKAT_FBFUSE_BEAMBANDPASSGENERATOR_HPP

#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/common.hpp"
#include <vector>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {

struct ChannelSums
{
    double sum;
    double sum_of_squares;
};

struct ChannelStatistics
{
    float mean;
    float variance;
};

/**
 * @brief      Class for coherent beamformer.
 */
template <typename Handler>
class BeamBandpassGenerator
{
public:
    /**
     * @brief      Constructs a BeamBandpassGenerator object.
     */
    BeamBandpassGenerator(
        unsigned int nbeams,
        unsigned int nchans_per_subband,
        unsigned int nsubbands,
        unsigned int heap_size,
        unsigned int nbuffer_acc,
        Handler& handler);
    ~BeamBandpassGenerator();
    BeamBandpassGenerator(BeamBandpassGenerator const&) = delete;

    void init(RawBytes&);
    bool operator()(RawBytes&);

private:
    
    unsigned int _nbeams;
    unsigned int _nchans_per_subband;
    unsigned int _nsubbands;
    unsigned int _heap_size;
    unsigned int _nbuffer_acc;
    Handler& _handler;
    std::size_t _naccumulated;
    std::size_t _count;
    std::vector<ChannelSums> _calculation_buffer;
    std::vector<ChannelStatistics> _output_buffer;
};

} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

#include "psrdada_cpp/meerkat/fbfuse/detail/BeamBandpassGenerator.cpp"

#endif //PSRDADA_CPP_MEERKAT_FBFUSE_BEAMBANDPASSGENERATOR_HPP
