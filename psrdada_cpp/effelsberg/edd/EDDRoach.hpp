#ifndef PSRDADA_CPP_EFFELSBERG_EDD_EDDPOLNMERGE_HPP
#define PSRDADA_CPP_EFFELSBERG_EDD_EDDPOLNMERGE_HPP
#include "psrdada_cpp/dada_write_client.hpp"
#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/common.hpp"
#include <vector>

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {

class EDDRoach
{
public:
    EDDRoach(std::size_t nsamps_per_heap, std::size_t npol, DadaWriteClient& writer);
    ~EDDRoach();

    /**
     * @brief      A callback to be called on connection
     *             to a ring buffer.
     *
     * @detail     The first available header block in the
     *             in the ring buffer is provided as an argument.
     *             It is here that header parameters could be read
     *             if desired.
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

private:
    std::size_t _nsamps_per_heap;
    std::size_t _npol;	
    DadaWriteClient& _writer;
};

} // edd
} // effelsberg
} // psrdada_cpp

#endif //PSRDADA_CPP_EFFELSBERG_EDD_EDDPOLNMERGE_HPP
