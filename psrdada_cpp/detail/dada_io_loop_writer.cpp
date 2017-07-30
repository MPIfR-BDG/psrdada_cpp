#include "psrdada_cpp/dada_io_loop_writer.hpp"

namespace psrdada_cpp
{
    template <class ApplicationType>
    DadaIoLoopWriter<ApplicationType>::DadaIoLoopWriter(key_t key, MultiLog& log)
    : DadaIoLoop(key, log)
    {
    }

    template <class ApplicationType>
    DadaIoLoopWriter<ApplicationType>::~DadaIoLoopWriter()
    {
    }


    template <class ApplicationType>
    void DadaIoLoopWriter<ApplicationType>::run()
    {
        if (_running)
        {
            throw std::runtime_error("IO loop is already running");
        }
        _running = true;
        while (!_stop)
        {
            BOOST_LOG_TRIVIAL(info) << "Attaching new write client to buffer";
            DadaWriteClient client(_key,_log);
            auto& header_stream = client.header_stream();
            static_cast<ApplicationType*>(this)->on_connect(header_stream.next());
            header_stream.release();
            auto& data_stream = client.data_stream();
            while (!_stop)
            {
                bool eod = static_cast<ApplicationType*>(this)->on_next(data_stream.next());
                data_stream.release(eod);
                if (eod)
                {
                    BOOST_LOG_TRIVIAL(info) << "Final buffer written";
                    _stop = true;
                    break;
                }
            }
        }
        _running = false;
    }

} //namespace psrdada_cpp