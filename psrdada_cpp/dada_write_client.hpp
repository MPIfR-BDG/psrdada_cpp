#ifndef PSRDADA_CPP_DADA_WRITE_CLIENT_HPP
#define PSRDADA_CPP_DADA_WRITE_CLIENT_HPP

#include "psrdada_cpp/dada_client_base.hpp"
#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/common.hpp"

namespace psrdada_cpp {

    class DadaWriteClient: public DadaClientBase
    {
    public:
        class HeaderStream
        {
        private:
            DadaWriteClient& _parent;
            std::unique_ptr<RawBytes> _current_block;

        public:
            HeaderStream(DadaWriteClient& parent);
            HeaderStream(HeaderStream const&) = delete;
            ~HeaderStream();
            RawBytes& next();
            void release();
        };

        class DataStream
        {
        private:
            DadaWriteClient& _parent;
            std::unique_ptr<RawBytes> _current_block;
            std::size_t _block_idx;

        public:
            DataStream(DadaWriteClient& parent);
            DataStream(DataStream const&) = delete;
            ~DataStream();
            RawBytes& next();
            void release(bool eod=false);
            std::size_t block_idx() const;
        };

    public:
        DadaWriteClient(key_t key, MultiLog& log);
        DadaWriteClient(DadaWriteClient const&) = delete;
        ~DadaWriteClient();
        HeaderStream& header_stream();
        DataStream& data_stream();

    private:
        void lock();
        void release();
        bool _locked;
        HeaderStream _header_stream;
        DataStream _data_stream;
    };

} //namespace psrdada_cpp

#endif //PSRDADA_CPP_DADA_WRITE_CLIENT_HPP