#ifndef PSRDADA_CPP_DADA_OUTPUT_STREAM_HPP
#define PSRDADA_CPP_DADA_OUTPUT_STREAM_HPP

#include "psrdada_cpp/dada_write_client.hpp"
#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/common.hpp"

namespace psrdada_cpp
{
    class DadaOutputStream
    {
    public:
        DadaOutputStream(key_t key, MultiLog& log);
        ~DadaOutputStream();
        void init(RawBytes&);
        void operator()(RawBytes&);
        DadaWriteClient const& client() const;

    private:
        DadaWriteClient _writer;
    };
}

#endif //PSRDADA_CPP_DADA_OUTPUT_STREAM_HPP
