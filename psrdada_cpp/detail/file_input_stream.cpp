#include "psrdada_cpp/file_input_stream.hpp"
#include "psrdada_cpp/sigprocheader.hpp"

#include <chrono>
#include <thread>

namespace psrdada_cpp {

    template <class HandlerType>
    FileInputStream<HandlerType>::FileInputStream(std::string fileName, std::size_t headersize, std::size_t nbytes, HandlerType& handler, float streamtime)
    : _headersize(headersize)
    , _nbytes(nbytes)
    , _streamtime(streamtime)
    , _handler(handler)
    , _stop(false)
    , _running(false)
    {
        const char *filename = fileName.c_str();
        _filestream.exceptions(std::ifstream::badbit);
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

        SigprocHeader sigheader;
        FilHead filheader = {"","",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        _filestream.seekg(0, _filestream.beg);
        sigheader.read_header(_filestream, filheader);

        std::size_t filesize = 0;
        _filestream.seekg(0, _filestream.end);
        std::size_t fullsize = _filestream.tellg();
        filesize = fullsize - _headersize;
        _filestream.seekg(_headersize, _filestream.beg);
        float filetime = static_cast<float>(filesize / (filheader.nbits / 8) / filheader.nchans) * filheader.tsamp; 

        std::chrono::time_point<std::chrono::steady_clock> streamstart;
        float streamed = 0.0f;
        int streamcount = 0;
        // Continue to stream data until the end
        streamstart = std::chrono::steady_clock::now();
        while (!_stop)
        {
            BOOST_LOG_TRIVIAL(info) << "Reading data from the file";
            // Read data from file here
            RawBytes data_block(data_ptr, _nbytes, 0, false);
            while (!_stop)
            {
                if (_filestream.eof())
                {
                    streamed += filetime;
                    streamcount++;
                    BOOST_LOG_TRIVIAL(info) << "Reached end of file";
                    if (streamed >= _streamtime) {
                    BOOST_LOG_TRIVIAL(info) << "Closing the stream";
                    BOOST_LOG_TRIVIAL(info) << "Streamed a total of " << streamed << "s in " << streamcount << " iterations";
                        _filestream.close();
                        break;
                    }
                    BOOST_LOG_TRIVIAL(info) << "Repeating the file again";
                    _filestream.clear();
                    _filestream.seekg(_headersize, _filestream.beg);
                    // TODO: Implement a better, proper streaming behaviour
                    std::this_thread::sleep_until(streamstart + std::chrono::seconds(static_cast<int>(streamcount * filetime)));
                }
                _filestream.read(data_ptr, _nbytes);
                data_block.used_bytes(data_block.total_bytes());
                _handler(data_block);
            }
            _stop = true;
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
