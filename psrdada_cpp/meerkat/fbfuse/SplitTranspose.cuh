#ifndef PSRDADA_CPP_MEERKAT_FBFUSE_SPLITTRANSPOSE_HPP
#define PSRDADA_CPP_MEERKAT_FBFUSE_SPLITTRANSPOSE_HPP

#include "psrdada_cpp/meerkat/fbfuse/PipelineConfig.hpp"
#include "psrdada_cpp/common.hpp"
#include "thrust/device_vector.h"

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace kernel {

/**
 * @brief      Perform a split transpose of input data
 *             in TAFTP order.
 *
 * @param      input            8-bit complex voltages data in TAFTP order
 * @param      output           8-bit real power data in FTPA order
 * @param[in]  total_nantennas  The total number of antennas (e.g. T[A]FTP)
 * @param[in]  used_nantennas   The number of antennas in the split subset
 * @param[in]  start_antenna    The index of the first antenna in the split subset
 * @param[in]  nchans           The number of frequency channels (e.g. TA[F]TP)
 * @param[in]  ntimestamps      The number of timestamps (outer T dimension, e.g. [T]AFTP)
 */
__global__
void split_transpose_k(
    char2 const * __restrict__ input,
    char2 * __restrict__ output,
    int total_nantennas,
    int used_nantennas,
    int start_antenna,
    int nchans,
    int ntimestamps);

} //namespace kernel

/**
 * @brief      Class for split transposing voltate data
 *
 * @detail     This class wraps a split transpose method that
 *             is used to convert from the TAFTP order data
 *             received by FBFUSE into FTPA order data that
 *             can be most efficiently beamformed. The reason
 *             we call this a "split" transpose is that not all
 *             antennas are kept in the output (i.e. the two
 *             "A" dimensions are not the same in input and
 *             output). The primiary limitation here is that
 *             subset of antennas extracted in the split
 *             transpose MUST BE CONTIGUOUS.
 */
class SplitTranspose
{
public:
    typedef thrust::device_vector<char2> VoltageType;

public:
    /**
     * @brief      Create a new split transposer
     *
     * @param      config  The pipeline configuration
     */
    explicit SplitTranspose(PipelineConfig const& config);
    ~SplitTranspose();
    SplitTranspose(SplitTranspose const&) = delete;

    /**
     * @brief      Perform a split transpose on the data
     *
     * @param      taftp_voltages  The input TAFTP voltages
     * @param      ftpa_voltages   The output FTPA voltages
     * @param[in]  stream          The cuda stream to use
     */
    void transpose(VoltageType const& taftp_voltages,
        VoltageType& ftpa_voltages, cudaStream_t stream);

private:
    PipelineConfig const& _config;
    std::size_t _heap_group_size;
    std::size_t _output_size_per_heap_group;

};

} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

#endif //PSRDADA_CPP_MEERKAT_FBFUSE_SPLITTRANSPOSE_HPP