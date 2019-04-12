#include "psrdada_cpp/dada_input_stream.hpp"

namespace psrdada_cpp {

    template <class HandlerType>
    DadaInputStream<HandlerType>::DadaInputStream(
        DadaReadClient& client,
        HandlerType& handler)
    : _client(client)
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
        bool handler_stop_request = false;
        if (_running)
        {
            throw std::runtime_error("Stream is already running");
        }
        _running = true;
        BOOST_LOG_TRIVIAL(info) << "Fetching header";
        auto& header_stream = _client.header_stream();
        auto& header_block = header_stream.next();
        _handler.init(header_block);
        header_stream.release();
        auto& data_stream = _client.data_stream();
        while (!_stop && !handler_stop_request)
        {
            if (data_stream.at_end())
            {
                BOOST_LOG_TRIVIAL(info) << "Reached end of data";
                break;
            }
            handler_stop_request = _handler(data_stream.next());
            data_stream.release();
        }
        _running = false;
    }

    template <class HandlerType>
    void DadaInputStream<HandlerType>::stop()
    {
        _stop = true;
    }

} //psrdada_cpp
