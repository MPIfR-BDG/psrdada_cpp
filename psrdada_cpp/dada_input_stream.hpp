#ifndef PSRDADA_CPP_DADA_INPUT_STREAM_HPP
#define PSRDADA_CPP_DADA_INPUT_STREAM_HPP

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

    template <class HandlerType>
    DadaInputStream<HandlerType>::DadaInputStream(key_t key, MultiLog& log, HandlerType& handler)
    : _key(key)
    , _log(log)
    , _handler(handler)
    , _stop(false)
    , _running(false)
    {
    }

    template <class HandlerType>
    DadaInputStream<HandlerType>::~DadaInputStream()
    {
    }

    template <class HandlerType>
    void DadaInputStream<HandlerType>::start()
    {
        if (_running)
        {
            throw std::runtime_error("Stream is already running");
        }
        _running = true;
        while (!_stop)
        {
            BOOST_LOG_TRIVIAL(info) << "Attaching new read client to buffer";
            DadaReadClient client(_key,_log);
            auto& header_stream = client.header_stream();
            _handler.init(header_stream.next());
            header_stream.release();
            auto& data_stream = client.data_stream();
            while (!_stop)
            {
                if (data_stream.at_end())
                {
                    BOOST_LOG_TRIVIAL(info) << "Reached end of data";
                    break;
                }
                _handler(data_stream.next());
                data_stream.release();
            }
        }
        _running = false;
    }

    template <class HandlerType>
    void DadaInputStream<HandlerType>::stop()
    {
        _stop = true;
    }


}

#endif //PSRDADA_CPP_DADA_INPUT_STREAM_HPP