#ifndef PSRDADA_CPP_SIMPLE_SHM_WRITER_HPP
#define PSRDADA_CPP_SIMPLE_SHM_WRITER_HPP

#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/common.hpp"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <cstring>
#include <sys/mman.h>
#include <sstream>

namespace psrdada_cpp {

    class SimpleShmWriter
    {
    public:
        explicit SimpleShmWriter(
            std::string const& shm_key,
            std::size_t header_size,
            std::size_t data_size);
        SimpleShmWriter(SimpleShmWriter const&) = delete;
        ~SimpleShmWriter();
        void init(RawBytes&);
        bool operator()(RawBytes&);

    private:
        std::string const _shm_key;
        std::size_t _header_size;
        std::size_t _data_size;
        int _shm_fd;
        void* _shm_ptr;
    };
}//psrdada_cpp

#endif //PSRDADA_CPP_SIMPLE_SHM_WRITER_HPP
