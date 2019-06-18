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
    BOOST_LOG_TRIVIAL(debug) << "Initialising BeamBandpassGenerator instance";
    BOOST_LOG_TRIVIAL(debug) << "Resizing output buffer to "
                             << nchans_per_subband * nsubbands * nbeams
                             << " bytes";
    _calculation_buffer.resize(nchans_per_subband * nsubbands * nbeams);
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

    BOOST_LOG_TRIVIAL(debug) << "Received " << block.used_bytes()
                             << " bytes data block";
    BOOST_LOG_TRIVIAL(debug) << "Determined block dimensions (TBFTF order): "
                             << "[" << nheap_groups << ", "
                             << _nbeams << ", "
                             << _nsubbands << ", "
                             << nsamps_per_heap << ", "
                             << _nchans_per_subband << "]";
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
                        auto& sums = _calculation_buffer[output_idx_2];
                        sums.sum += value;
                        sums.sum_of_squares += value * value;
                    }
                }
            }
        }
    }
    ++_naccumulated;
    _count += total_nsamps;
    BOOST_LOG_TRIVIAL(debug) << "Nblocks accumulated = " << _naccumulated;
    BOOST_LOG_TRIVIAL(debug) << "Total samples accumulated = " << _count;
    if (_naccumulated >= _nbuffer_acc)
    {
        BOOST_LOG_TRIVIAL(debug) << "Accumulation threshold met, outputing statistics";
        // Convert statistics to true mean and variance

        for (std::size_t ii = 0; ii < _output_buffer.size(); ++ii)
        {
            auto const& sums = _calculation_buffer[ii];
            auto& stats = _output_buffer[ii];
            double mean = sums.sum / _count;
            double variance = (sums.sum_of_squares / _count) - (mean * mean);
            stats.mean = static_cast<float>(mean);
            stats.variance = static_cast<float>(variance);
        }

        // Call handler
        const std::size_t nbytes = _output_buffer.size() * sizeof(ChannelStatistics);
        RawBytes bytes(reinterpret_cast<char*>(_output_buffer.data()), nbytes, nbytes);
        _handler(bytes);
        std::fill(_calculation_buffer.begin(), _calculation_buffer.end(), {0.0, 0.0});
        _count = 0;
        _naccumulated = 0;
    }
    return false;
}

} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp
