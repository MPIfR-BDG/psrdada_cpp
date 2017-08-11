#ifndef PSRDADA_CPP_DADA_INPUT_STREAM_HPP
#define PSRDADA_CPP_DADA_INPUT_STREAM_HPP

#include "psrdada_cpp/dada_read_client.hpp"
#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/common.hpp"

namespace psrdada_cpp
{

    template <class HandlerType>
    class DadaInputStream
    {
    public:
        DadaInputStream(key_t key, MultiLog& log, HandlerType& handler);
        ~DadaInputStream();
        void start();
        void stop();

    private:
        key_t _key;
        MultiLog& _log;
        HandlerType& _handler;
        bool _stop;
        bool _running;
    };
} //psrdada_cpp

#include "psrdada_cpp/detail/dada_input_stream.cpp"

#endif //PSRDADA_CPP_DADA_INPUT_STREAM_HPP