#ifndef PSRDADA_CPP_FILE_INPUT_STREAM_HPP
#define PSRDADA_CPP_FILE_INPUT_STREAM_HPP
#include <fstream>
#include <cstdlib>
#include "psrdada_cpp/raw_bytes.hpp"
/**
 * @detail: A simple file input stream. Will go through one entire file.
 * Will assume there is some amount of header to the file.
 */


namespace psrdada_cpp
{

    template <class HandlerType>
    class FileInputStream
    {
    public:
        FileInputStream(std::string filename, std::size_t headersize, std::size_t nbytes, HandlerType& handler);
        ~FileInputStream();
        void start();
        void stop();

    private:
        std::size_t _headersize;
        std::size_t _nbytes;
        std::ifstream _filestream;
        HandlerType& _handler;
        bool _stop;
        bool _running;
    };
} //psrdada_cpp

#include "psrdada_cpp/detail/file_input_stream.cpp"

#endif //PSRDADA_CPP_FILE_INPUT_STREAM_HPP
