#include "psrdada_cpp/dada_input_stream.hpp"

namespace psrdada_cpp {

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
                _stop = _handler(data_stream.next());
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

} //psrdada_cpp