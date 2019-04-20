#ifndef PSRDADA_CPP_SIMPLE_FILE_WRITER_HPP
#define PSRDADA_CPP_SIMPLE_FILE_WRITER_HPP

#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/common.hpp"
#include <fstream>
#include <iomanip>

namespace psrdada_cpp {

    class SimpleFileWriter
    {
    public:
        explicit SimpleFileWriter(std::string filename);
        SimpleFileWriter(SimpleFileWriter const&) = delete;
        ~SimpleFileWriter();
        void init(RawBytes&);
        void init(RawBytes&, std::size_t);
        bool operator()(RawBytes&);

    private:
        std::ofstream _outfile;
    };
}//psrdada_cpp

#endif //PSRDADA_CPP_SIMPLE_FILE_WRITER_HPP
