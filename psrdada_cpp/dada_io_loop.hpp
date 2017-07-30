#ifndef PSRDADA_CPP_DADA_IO_LOOP_HPP
#define PSRDADA_CPP_DADA_IO_LOOP_HPP

#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/common.hpp"

namespace psrdada_cpp
{

    class DadaIoLoop
    {
    protected:
        key_t _key;
        MultiLog& _log;
        bool _stop;
        bool _running;

    public:
        DadaIoLoop(key_t key, MultiLog& log);
        DadaIoLoop(DadaIoLoop const&) = delete;
        ~DadaIoLoop();
        void stop();
        virtual void run()=0;
    };

} //namespace psrdada_cpp

#endif //PSRDADA_CPP_DADA_IO_LOOP_HPP