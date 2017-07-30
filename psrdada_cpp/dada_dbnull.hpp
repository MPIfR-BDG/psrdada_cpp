#ifndef PSRDADA_CPP_DADA_DBNULL_HPP
#define PSRDADA_CPP_DADA_DBNULL_HPP

#include "psrdada_cpp/dada_io_loop_reader.hpp"
#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/common.hpp"

namespace psrdada_cpp
{
    /**
     * @brief      Class for reading from a DADA buffer
     *             and doing nothing with the data.
     *
     * @detail     This class is intended as both an example
     *             for how to use CRTP with DadaIoLoopReader
     *             and as a tool for testing.
     */
    class DbNull
    : public DadaIoLoopReader<DbNull>
    {
    private:
        std::size_t _nbytes;
        bool _infinite;

    public:
        /**
         * @brief      Create a new instance
         *
         * @param[in]  key     The shared memory key
         * @param      log     A MultiLog instance
         * @param[in]  nbytes  The number of bytes to read from the buffer.
         *                     Setting this value to 0 will cause the instance
         *                     to read forever.
         */
        DbNull(key_t key, MultiLog& log, std::size_t nbytes);
        ~DbNull();

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
        void on_connect(RawBytes& block);

        /**
         * @brief      A callback to be called on acqusition of a new
         *             data block.
         *
         * @param      block  A RawBytes object wrapping a DADA data buffer
         */
        void on_next(RawBytes& block);
    };

} //namespace psrdada_cpp
#endif //PSRDADA_CPP_DADA_DBNULL_HPP