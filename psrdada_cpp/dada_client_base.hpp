#ifndef PSRDADA_CPP_DADA_CLIENT_BASE_HPP
#define PSRDADA_CPP_DADA_CLIENT_BASE_HPP

#include "dada_hdu.h"
#include "dada_def.h"
#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/common.hpp"

namespace psrdada_cpp {

    class DadaClientBase
    {
    private:
        key_t _key;

    protected:
        dada_hdu_t* _hdu;
        bool _connected;
        MultiLog& _log;

    public:
        DadaClientBase(key_t key, MultiLog& log);
        DadaClientBase(DadaClientBase const&) = delete;
        ~DadaClientBase();
        std::size_t data_buffer_size();
        std::size_t header_buffer_size();
        std::size_t data_buffer_count();
        std::size_t header_buffer_count();

    private:
        void connect();
        void disconnect();
};

} //namespace psrdada_cpp

#endif //PSRDADA_CPP_DADA_CLIENT_BASE_HPP