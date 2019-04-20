#ifndef PSRDADA_CPP_TRANSPOSE_TO_DADA_HPP
#define PSRDADA_CPP_TRANSPOSE_TO_DADA_HPP

#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/common.hpp"

namespace psrdada_cpp {
namespace meerkat {
namespace tuse {

namespace transpose{

    /**
     * @brief the method that does the actual transpose
     */
    void do_transpose(RawBytes& transposed_data, RawBytes& input_data, std::uint32_t nchans, std::uint32_t nsamples, std::uint32_t nfreq, std::uint32_t beamnum, std::uint32_t numbeams, std::uint32_t ngroups);
}

template <class HandlerType>
class TransposeToDada
{

public:
    TransposeToDada(std::size_t numbeams, std::vector<std::shared_ptr<HandlerType>> handler);
    ~TransposeToDada();

    /**
     * @brief      A transpose method to be called on connection
     *             to a ring buffer.
     *
     * @detail     The number of beams to process and a vector of
     * 		   shared pointers to open DADA blocks are given
     * 		   as arguments. The transpose is performed on
     * 		   a beam to beam basis.
     *
     * @param      block  A RawBytes object wrapping a DADA header buffer
     */
    void init(RawBytes& block);

    /**
     * @brief      A callback to be called on acqusition of a new
     *             data block.
     *
     * @param      block  A RawBytes object wrapping a DADA data buffer
     */
    bool operator()(RawBytes& block);


	/**
 	 *@brief: Setter for number of beams
	 */

	void set_nbeams(const int nbeams);

	/**
 	 *@brief: Setter for ngroups
	 */

	void set_ngroups(const int ngroups);

    /**
     * @brief Setter for frequency channels
     */

    void set_nchans(const int nchans);

    /**
     * @brief Setter of number of time samples
     */

    void set_nsamples(const int nsamples);

    /**
     * @brief Setter for number of frequency blocks
     */

    void set_nfreq(const int _nfreq);

    /**
     * @brief getter for number of channels
     */

    std::uint32_t nchans();

    /**
     * @brief getter for number of time samples
     */

    std::uint32_t nsamples();

    /**
     * @brief getter for frequency blocks
     */

    std::uint32_t nfreq();

    /**
     *@brief: getter for ngroups
     */

    std::uint32_t ngroups();

    /**
     *@brief: getter for nbeams
     */

    std::uint32_t nbeams();

private:
    std::uint32_t _numbeams;
    std::vector<std::shared_ptr<HandlerType>> _handler;
    std::uint32_t _nchans;
    std::uint32_t _nsamples;
    std::uint32_t _nfreq;
    std::uint32_t _ngroups;

};

} // namespace tuse
} // namespace meerkat
} // namespace psrdada_cpp

#include "psrdada_cpp/meerkat/tuse/detail/transpose_to_dada.cpp"
#endif //PSRDADA_CPP_TRANSPOSE_TO_DADA_HPP
