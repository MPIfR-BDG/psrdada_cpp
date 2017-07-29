#ifndef PSRDADA_CPP_DADA_WRITE_CLIENT_HPP
#define PSRDADA_CPP_DADA_WRITE_CLIENT_HPP

#include "psrdada_cpp/dada_client_base.hpp"
#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/common.hpp"

namespace psrdada_cpp {

    class DadaWriteClient: public DadaClientBase
    {
    private:
        bool _locked;
        std::unique_ptr<RawBytes> _current_header_block;
        std::unique_ptr<RawBytes> _current_data_block;

    public:
        DadaWriteClient(key_t key, MultiLog& log);
        DadaWriteClient(DadaWriteClient const&) = delete;
        ~DadaWriteClient();
        RawBytes& acquire_header_block();
        void release_header_block();
        RawBytes& acquire_data_block();
        void release_data_block(bool eod=false);

    private:
        void lock();
        void release();
    };

} //namespace psrdada_cpp

#endif //PSRDADA_CPP_DADA_WRITE_CLIENT_HPP