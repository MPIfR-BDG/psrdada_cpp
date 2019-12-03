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
        FilHead filheader;
        _filestream.seekg(0, _filestream.beg);
        sigheader.read_header(_filestream, filheader);

        if ( _nbytes/(filheader.nbits/8) % filheader.nchans != 0) {
            BOOST_LOG_TRIVIAL(info) << "nbits:" << filheader.nbits ;
            throw std::logic_error("Number of samples to stream to a block has to be a multiple of the number of channels");
        }

        std::size_t filesize = 0;
        _filestream.seekg(0, _filestream.end);
        filesize = static_cast<std::size_t>(_filestream.tellg()) - _headersize;
        _filestream.seekg(_headersize, _filestream.beg);
        float blocktime = static_cast<float>(_nbytes / (filheader.nbits / 8) / filheader.nchans) * filheader.tsamp;

        std::chrono::time_point<std::chrono::steady_clock> streamstart;
        float streamed = 0.0f;
        int streamcount = 0;
        // Continue to stream data until the end
        streamstart = std::chrono::steady_clock::now();
        while (!_stop) {
            BOOST_LOG_TRIVIAL(info) << "Reading data from the file";
            BOOST_LOG_TRIVIAL(info) << "Streaming " << _nbytes << "B every " << blocktime << "s (" << static_cast<float>(_nbytes) / 1024.0f / 1024.0f << "MiBps";
            // Read data from file here
            RawBytes data_block(data_ptr, _nbytes, 0, false);
            while (!_stop)
            {
                if ((static_cast<std::size_t>(_filestream.tellg()) + _nbytes) > filesize) {
                    BOOST_LOG_TRIVIAL(info) << "Will read beyond the end of file";
                    BOOST_LOG_TRIVIAL(info) << "Going to the start of the file again";
                    _filestream.clear();
                    _filestream.seekg(_headersize, _filestream.beg);
                }

                _filestream.read(data_ptr, _nbytes);
                data_block.used_bytes(data_block.total_bytes());
                _handler(data_block);

                streamed += blocktime;
                streamcount++;
                if (streamed >= _streamtime) {
                    BOOST_LOG_TRIVIAL(info) << "Closing the stream";
                    BOOST_LOG_TRIVIAL(info) << "Streamed a total of " << streamed << "s in " << streamcount << " iterations";
                    _filestream.close();
                    _stop = true;
                    break;
                }
                //std::this_thread::sleep_until(streamstart + std::chrono::milliseconds(static_cast<int>(streamcount * blocktime * 1000.0f)));
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
