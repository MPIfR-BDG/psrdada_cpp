#ifndef PSRDADA_CPP_DADA_JUNKDB_HPP
#define PSRDADA_CPP_DADA_JUNKDB_HPP

#include "psrdada_cpp/dada_io_loop_writer.hpp"
#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/common.hpp"

namespace psrdada_cpp
{
    class JunkDb
    : public DadaIoLoopWriter<JunkDb>
    {
    private:
        std::size_t _nbytes;
        bool _infinite;

    public:
        JunkDb(key_t key, MultiLog& log, std::size_t nbytes);
        ~JunkDb();
        void on_connect(RawBytes& block);
        bool on_next(RawBytes& block);
    };

} //namespace psrdada_cpp
#endif //PSRDADA_CPP_DADA_JUNKDB_HPP