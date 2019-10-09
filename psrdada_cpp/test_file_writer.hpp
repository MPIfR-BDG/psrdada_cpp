#ifndef PSRDADA_CPP_TEST_FILE_WRITER_HPP
#define PSRDADA_CPP_TEST_FILE_WRITER_HPP

#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/sigprocheader.hpp"
#include <fstream>
#include <iomanip>

namespace psrdada_cpp {

/* 
 * @brief: A file writer for testing purposes. Writes out files of specified size
 */

    class TestFileWriter
    {
    public:
        explicit TestFileWriter(std::string filename, std::size_t filesize);

        TestFileWriter(TestFileWriter const&) = delete;
        ~TestFileWriter();
        void init(RawBytes&);
        bool operator()(RawBytes&);

    private:
        std::string _basefilename;
        char* _header;
        std::size_t _filesize;
        std::size_t _filenum;
        std::size_t _wsize;
        std::ofstream _outfile;
        std::size_t _header_size;
    };
}//psrdada_cpp

#endif //PSRDADA_CPP_TEST_FILE_WRITER_HPP
