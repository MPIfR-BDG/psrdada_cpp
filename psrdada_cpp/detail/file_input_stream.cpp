#include "psrdada_cpp/file_input_stream.hpp"

namespace psrdada_cpp {

    template <class HandlerType>
    FileInputStream<HandlerType>::FileInputStream(std::string fileName, std::size_t headersize, std::size_t nbytes, HandlerType& handler)
    : _headersize(headersize)
    , _nbytes(nbytes)
    , _handler(handler)
    , _stop(false)
    , _running(false)
    {
        const char *filename = fileName.c_str();
        _filestream.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        _filestream.open(filename, std::ifstream::in | std::ifstream::binary);
        if (!_filestream.is_open())
        {
            throw std::runtime_error("File could not be opened");
        }
    }

    template <class HandlerType>
    FileInputStream<HandlerType>::~FileInputStream()
    {
        _filestream.close();
    }

    template <class HandlerType>
    void FileInputStream<HandlerType>::start()
    {
        if (_running)
        {
            throw std::runtime_error("Stream is already running");
        }
        _running = true;
        
        // Get the header
        char* header_ptr = new char[4096];
        char* data_ptr = new char[_nbytes];
        RawBytes header_block(header_ptr, 4096, 0, false);
        _filestream.read(header_ptr, _headersize);
        header_block.used_bytes(4096);
        _handler.init(header_block);

        // Continue to stream data until the end
        while (!_stop)
        {
            BOOST_LOG_TRIVIAL(info) << "Reading data from the file";
            // Read data from file here
            RawBytes data_block(data_ptr, _nbytes, 0, false);
            while (!_stop)
            {
                if (_filestream.eof())
                {
                    BOOST_LOG_TRIVIAL(info) << "Reached end of file";
                    _filestream.close();
                    break;
                }
                _filestream.read(data_ptr, _nbytes);
                data_block.used_bytes(data_block.total_bytes());
                _handler(data_block);
            }
            data_block.~RawBytes();
        }
        _running = false;
    }

    template <class HandlerType>
    void FileInputStream<HandlerType>::stop()
    {
        _stop = true;
    }

} //psrdada_cpp
