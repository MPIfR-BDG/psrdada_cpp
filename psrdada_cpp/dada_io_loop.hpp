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

        /**
         * @brief      Create new instance
         *
         * @param[in]  key   A hexadecimal shared memory key
         * @param      log   A MultiLog instance
         */
        DadaIoLoop(key_t key, MultiLog& log);
        DadaIoLoop(DadaIoLoop const&) = delete;
        ~DadaIoLoop();

        /**
         * @brief      Stop the IO loop processing
         */
        void stop();

        /**
         * @brief      Start the IO loop processing
         */
        virtual void run()=0;
    };

} //namespace psrdada_cpp

#endif //PSRDADA_CPP_DADA_IO_LOOP_HPP