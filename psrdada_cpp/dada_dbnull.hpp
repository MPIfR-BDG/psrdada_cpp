#ifndef PSRDADA_CPP_DADA_DBNULL_HPP
#define PSRDADA_CPP_DADA_DBNULL_HPP

#include "psrdada_cpp/dada_io_loop_reader.hpp"
#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/common.hpp"

namespace psrdada_cpp
{
    class DbNull
    : public DadaIoLoopReader<DbNull>
    {
    private:
        std::size_t _nbytes;
        bool _infinite;

    public:
        DbNull(key_t key, MultiLog& log, std::size_t nbytes);
        ~DbNull();
        void on_connect(RawBytes& block);
        void on_next(RawBytes& block);
    };

} //namespace psrdada_cpp
#endif //PSRDADA_CPP_DADA_DBNULL_HPP