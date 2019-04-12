#ifndef PSRDADA_CPP_DADA_INPUT_STREAM_HPP
#define PSRDADA_CPP_DADA_INPUT_STREAM_HPP

#include "psrdada_cpp/dada_read_client.hpp"
#include "psrdada_cpp/common.hpp"

namespace psrdada_cpp
{

    template <class HandlerType>
    class DadaInputStream
    {
    public:
        DadaInputStream(DadaReadClient& client, HandlerType& handler);
        ~DadaInputStream();
        void start();
        void stop();

    private:
        DadaReadClient& _client;
        HandlerType& _handler;
        bool _stop;
        bool _running;
    };
} //psrdada_cpp

#include "psrdada_cpp/detail/dada_input_stream.cpp"

#endif //PSRDADA_CPP_DADA_INPUT_STREAM_HPP