#include "psrdada_cpp/meerkat/fbfuse/BeamBandpassGenerator.hpp"
#include <cassert>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {

template <typename Handler>
BeamBandpassGenerator<Handler>::BeamBandpassGenerator(
    unsigned int nbeams,
    unsigned int nchans_per_subband,
    unsigned int nsubbands,
    unsigned int heap_size,
    unsigned int nbuffer_acc,
    Handler& handler)
    : _nbeams(nbeams)
    , _nchans_per_subband(nchans_per_subband)
    , _nsubbands(nsubbands)
    , _heap_size(heap_size)
    , _nbuffer_acc(nbuffer_acc)
    , _handler(handler)
    , _naccumulated(0)
    , _count(0)
{
    _output_buffer.resize(nchans_per_subband * nsubbands * nbeams);
}

template <typename Handler>
BeamBandpassGenerator<Handler>::~BeamBandpassGenerator()
{

}

template <typename Handler>
void BeamBandpassGenerator<Handler>::init(RawBytes& block)
{
    _handler.init(block);
}

template <typename Handler>
bool BeamBandpassGenerator<Handler>::operator()(RawBytes& block)
{
    const std::size_t heap_group_size = _nbeams * _nsubbands * _heap_size;
    assert(block.used_bytes() % heap_group_size == 0);
    const std::size_t nheap_groups = block.used_bytes() / heap_group_size;
    const unsigned int f = _nchans_per_subband;
    const unsigned int tf = _heap_size;
    const unsigned int ftf = _nsubbands * tf;
    const unsigned int bftf = _nbeams * ftf;
    const unsigned int nsamps_per_heap = _heap_size / _nchans_per_subband;
    const unsigned int total_nchans = _nchans_per_subband * _nsubbands;
    const std::size_t total_nsamps = nheap_groups * nsamps_per_heap;

    for (unsigned int heap_group_idx = 0; heap_group_idx < nheap_groups; ++heap_group_idx)
    {
        std::size_t input_idx_0 = heap_group_idx * bftf;
        for (unsigned int beam_idx = 0; beam_idx < _nbeams; ++beam_idx)
        {
            std::size_t input_idx_1 = input_idx_0 + beam_idx * ftf;
            std::size_t output_idx_0 = beam_idx * total_nchans;
            for (unsigned int subband_idx = 0; subband_idx < _nsubbands; ++subband_idx)
            {
                std::size_t input_idx_2 = input_idx_1 + subband_idx * tf;
                std::size_t output_idx_1 = output_idx_0 + subband_idx * _nchans_per_subband;
                for (unsigned int samp_idx = 0; samp_idx < nsamps_per_heap; ++samp_idx)
                {
                    std::size_t input_idx_3 = input_idx_2 + samp_idx * f;
                    for (unsigned int chan_idx = 0; chan_idx < _nchans_per_subband; ++chan_idx)
                    {
                        std::size_t input_idx_4 = input_idx_3 + chan_idx;
                        std::size_t output_idx_2  = output_idx_1 + chan_idx;

                        float value = static_cast<float>(block.ptr()[input_idx_4]);
                        auto& stats = _output_buffer[output_idx_2];
                        stats.mean += value;
                        stats.variance += value * value;
                    }
                }
            }
        }
    }
    ++_naccumulated;
    _count += total_nsamps;
    if (_naccumulated >= _nbuffer_acc)
    {
        // Convert statistics to true mean and variance
        for (auto& stats: _output_buffer)
        {
            stats.mean /= _count;
            stats.variance = (stats.variance / _count) - (stats.mean * stats.mean);
        }

        // Call handler
        const std::size_t nbytes = _output_buffer.size() * sizeof(ChannelStatistics);
        RawBytes bytes(reinterpret_cast<char*>(_output_buffer.data()), nbytes, nbytes);
        _handler(bytes);

        // Clear buffers
        for (auto& stats: _output_buffer)
        {
            //stats.mean = 0.0;
            //stats.variance = 0.0f;
        }
        _count = 0;
        _naccumulated = 0;
    }
    return false;	
}

} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp
