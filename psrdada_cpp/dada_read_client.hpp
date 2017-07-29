#ifndef PSRDADA_CPP_DADA_READ_CLIENT_HPP
#define PSRDADA_CPP_DADA_READ_CLIENT_HPP

#include "psrdada_cpp/dada_client_base.hpp"
#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/common.hpp"

namespace psrdada_cpp {

    class DadaReadClient: public DadaClientBase
    {
    private:
        bool _locked;
        std::unique_ptr<RawBytes> _current_header_block;
        std::unique_ptr<RawBytes> _current_data_block;
        std::size_t _current_data_block_idx;

    public:
        DadaReadClient(key_t key, MultiLog& log);
        DadaReadClient(DadaReadClient const&) = delete;
        ~DadaReadClient();
        RawBytes& acquire_header_block();
        void release_header_block();
        RawBytes& acquire_data_block();
        void release_data_block();
        std::size_t current_data_block_idx() const;
        bool is_final_data_block() const;
        bool is_final_header_block() const;

    private:
        void lock();
        void release();
    };

} //namespace psrdada_cpp

#endif //PSRDADA_CPP_DADA_READ_CLIENT_HPP