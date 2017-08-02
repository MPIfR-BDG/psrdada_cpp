#ifndef PSRDADA_CPP_DADA_IO_LOOP_WRITER_HPP
#define PSRDADA_CPP_DADA_IO_LOOP_WRITER_HPP

#include "psrdada_cpp/dada_io_loop.hpp"
#include "psrdada_cpp/dada_write_client.hpp"
#include "psrdada_cpp/common.hpp"

namespace psrdada_cpp
{
    template <class ApplicationType>
    class DadaIoLoopWriter: public DadaIoLoop
    {
    public:
        /**
         * @brief      Create a IO loop for writing to a DADA buffer
         *
         * @param[in]  key   The hexidecimal shared memory key
         * @param      log   A MultiLog instance for logging buffer transactions
         */
        DadaIoLoopWriter(key_t key, MultiLog& log);
        ~DadaIoLoopWriter();

        /**
         * @brief      Start the IO loop processing
         */
        void run();
    };
} //namespace psrdada_cpp

#include "psrdada_cpp/detail/dada_io_loop_writer.cpp"
#endif //PSRDADA_CPP_DADA_IO_LOOP_WRITER_HPP