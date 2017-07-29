#ifndef PSRDADA_CPP_DADA_READ_CLIENT_HPP
#define PSRDADA_CPP_DADA_READ_CLIENT_HPP

#include "psrdada_cpp/dada_client_base.hpp"
#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/common.hpp"

namespace psrdada_cpp {

    class DadaReadClient: public DadaClientBase
    {
    public:
        class HeaderStream
        {
        private:
            DadaReadClient& _parent;
            std::unique_ptr<RawBytes> _current_block;

        public:
            HeaderStream(DadaReadClient& parent);
            HeaderStream(HeaderStream const&) = delete;
            ~HeaderStream();
            RawBytes& next();
            void release();
            bool at_end() const;
        };

        class DataStream
        {
        private:
            DadaReadClient& _parent;
            std::unique_ptr<RawBytes> _current_block;
            std::size_t _block_idx;

        public:
            DataStream(DadaReadClient& parent);
            DataStream(DataStream const&) = delete;
            ~DataStream();
            RawBytes& next();
            void release();
            bool at_end() const;
            std::size_t block_idx() const;
        };

    private:
        bool _locked;
        HeaderStream _header_stream;
        DataStream _data_stream;

    public:
        DadaReadClient(key_t key, MultiLog& log);
        DadaReadClient(DadaReadClient const&) = delete;
        ~DadaReadClient();
        HeaderStream& header_stream();
        DataStream& data_stream();

    private:
        void lock();
        void release();
    };

} //namespace psrdada_cpp

#endif //PSRDADA_CPP_DADA_READ_CLIENT_HPP