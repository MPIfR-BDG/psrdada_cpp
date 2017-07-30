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
            BOOST_LOG_TRIVIAL(debug) << "Header buffer is " << client.header_buffer_count()
            << " x " << client.header_buffer_size() << " bytes";
            BOOST_LOG_TRIVIAL(debug) << "Data buffer is " << client.data_buffer_count()
            << " x " << client.data_buffer_size() << " bytes";
            auto& header_stream = client.header_stream();
            auto& block = header_stream.next();
            BOOST_LOG_TRIVIAL(debug) << "Acquired header block ("
            << block.used_bytes() <<"/"<<block.total_bytes() << " bytes)";
            static_cast<ApplicationType*>(this)->on_connect(block);
            header_stream.release();
            BOOST_LOG_TRIVIAL(debug) << "Released header block";
            auto& data_stream = client.data_stream();
            while (!_stop)
            {
                if (data_stream.at_end())
                {
                    BOOST_LOG_TRIVIAL(info) << "Reached end of data";
                    break;
                }
                auto& data_block = data_stream.next();
                BOOST_LOG_TRIVIAL(debug) << "Acquired data block ("
                << block.used_bytes() <<"/"<<block.total_bytes() << " bytes)";
                static_cast<ApplicationType*>(this)->on_next(data_block);
                data_stream.release();
                BOOST_LOG_TRIVIAL(debug) << "Released data block";
            }
        }
        _running = false;
    }

} //namespace psrdada_cpp