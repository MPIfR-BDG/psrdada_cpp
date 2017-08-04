#ifndef PSRDADA_CPP_DADA_CLIENT_BASE_HPP
#define PSRDADA_CPP_DADA_CLIENT_BASE_HPP

#include "dada_hdu.h"
#include "dada_def.h"
#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/common.hpp"

namespace psrdada_cpp {

    class DadaClientBase
    {
    protected:
        key_t _key;
        dada_hdu_t* _hdu;
        bool _connected;
        MultiLog& _log;

    public:
        /**
         * @brief      Create a new basic DADA client instance
         *
         * @param[in]  key   The hexidecimal shared memory key
         * @param      log   A MultiLog instance for logging buffer transactions
         */
        DadaClientBase(key_t key, MultiLog& log);
        DadaClientBase(DadaClientBase const&) = delete;
        ~DadaClientBase();

        /**
         * @brief      Get the sizes of each data block in the ring buffer
         */
        std::size_t data_buffer_size() const;

        /**
         * @brief      Get the sizes of each header block in the ring buffer
         */
        std::size_t header_buffer_size() const;

        /**
         * @brief      Get the number of data blocks in the ring buffer
         */
        std::size_t data_buffer_count() const;

        /**
         * @brief      Get the number of header blocks in the ring buffer
         */
        std::size_t header_buffer_count() const;

        /**
         * @brief      Connect to ring buffer
         */
        void connect();

        /**
         * @brief      Disconnect from ring buffer
         */
        void disconnect();

        /**
         * @brief      Reconnect to the ring buffer
         */
        void reconnect();
};

} //namespace psrdada_cpp

#endif //PSRDADA_CPP_DADA_CLIENT_BASE_HPP