#ifndef PSRDADA_CPP_DADA_IO_LOOP_READER_HPP
#define PSRDADA_CPP_DADA_IO_LOOP_READER_HPP

#include "psrdada_cpp/dada_io_loop.hpp"
#include "psrdada_cpp/dada_read_client.hpp"
#include "psrdada_cpp/common.hpp"

namespace psrdada_cpp
{
    template <class ApplicationType>
    class DadaIoLoopReader: public DadaIoLoop
    {
    public:
        DadaIoLoopReader(key_t key, MultiLog& log);
        ~DadaIoLoopReader();
        void run();
    };
} //namespace psrdada_cpp

#include "psrdada_cpp/detail/dada_io_loop_reader.cpp"
#endif //PSRDADA_CPP_DADA_IO_LOOP_READER_HPP