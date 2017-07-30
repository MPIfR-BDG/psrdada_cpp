#include "psrdada_cpp/dada_io_loop_reader.hpp"

namespace psrdada_cpp
{
    template <class ApplicationType>
    DadaIoLoopReader<ApplicationType>::DadaIoLoopReader(key_t key, MultiLog& log)
    : DadaIoLoop(key, log)
    {
    }

    template <class ApplicationType>
    DadaIoLoopReader<ApplicationType>::~DadaIoLoopReader()
    {
    }


    template <class ApplicationType>
    void DadaIoLoopReader<ApplicationType>::run()
    {
        if (_running)
        {
            throw std::runtime_error("IO loop is already running");
        }
        _running = true;
        while (!_stop)
        {
            BOOST_LOG_TRIVIAL(info) << "Attaching new read client to buffer";
            DadaReadClient client(_key,_log);
            auto& header_stream = client.header_stream();
            static_cast<ApplicationType*>(this)->on_connect(header_stream.next());
            header_stream.release();
            auto& data_stream = client.data_stream();
            while (!_stop)
            {
                if (data_stream.at_end())
                {
                    BOOST_LOG_TRIVIAL(info) << "Reached end of data";
                    break;
                }
                static_cast<ApplicationType*>(this)->on_next(data_stream.next());
                data_stream.release();
            }
        }
        _running = false;
    }

} //namespace psrdada_cpp