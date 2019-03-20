#ifndef PSRDADA_CPP_DADA_READ_CLIENT_HPP
#define PSRDADA_CPP_DADA_READ_CLIENT_HPP

#include "psrdada_cpp/dada_client_base.hpp"
#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/common.hpp"

namespace psrdada_cpp {

    /**
     * @brief      Class that provides means for reading from
     *             a DADA ring buffer
     */
    class DadaReadClient: public DadaClientBase
    {
    public:
        class HeaderStream
        {
        private:
            DadaReadClient& _parent;
            std::unique_ptr<RawBytes> _current_block;

        public:
            /**
             * @brief      Create a new instance
             *
             * @param      parent  A reference to the parent reading client
             */
            HeaderStream(DadaReadClient& parent);
            HeaderStream(HeaderStream const&) = delete;
            ~HeaderStream();

            /**
             * @brief      Get the next header block in the ring buffer
             *
             * @detail     As only one block can be open at a time, release() must
             *             be called between subsequenct next() calls.
             *
             * @return     A RawBytes instance wrapping a pointer to share memory
             */
            RawBytes& next();

            /**
             * @brief      Release the current data block.
             *
             * @detail     This will mark the block as cleared, making it
             *             writeable by writing client.
             */
            void release();

            /**
             * @brief      Check if we have read the last header block in buffer.
             */
            bool at_end() const;

            void purge();
        };

        class DataStream
        {
        private:
            DadaReadClient& _parent;
            std::unique_ptr<RawBytes> _current_block;
            std::size_t _block_idx;

        public:
            /**
             * @brief      Create a new instance
             *
             * @param      parent  A reference to the parent reading client
             */
            DataStream(DadaReadClient& parent);
            DataStream(DataStream const&) = delete;
            ~DataStream();

            /**
             * @brief      Get the next data block in the ring buffer
             *
             * @detail     As only one block can be open at a time, release() must
             *             be called between subsequenct next() calls.
             *
             * @return     A RawBytes instance wrapping a pointer to share memory
             */
            RawBytes& next();

            /**
             * @brief      Release the current data block.
             *
             * @detail     This will mark the block as cleared, making it
             *             writeable by writing client.
             */
            void release();

            /**
             * @brief      Check if we have read the last data block in buffer.
             */
            bool at_end() const;

            void purge();

            /**
             * @brief      Return the index of the currently open block
             */
            std::size_t block_idx() const;
        };

    private:
        bool _locked;
        HeaderStream _header_stream;
        DataStream _data_stream;

    public:
        /**
         * @brief      Create a new client for reading from a DADA buffer
         *
         * @param[in]  key   The hexidecimal shared memory key
         * @param      log   A MultiLog instance for logging buffer transactions
         */
        DadaReadClient(key_t key, MultiLog& log);
        DadaReadClient(DadaReadClient const&) = delete;
        ~DadaReadClient();

        /**
         * @brief      Get a reference to a header stream manager
         *
         * @return     A HeaderStream manager object for the current buffer
         */
        HeaderStream& header_stream();

        /**
         * @brief      Get a reference to a data stream manager
         *
         * @return     A DataStream manager object for the current buffer
         */
        DataStream& data_stream();

        void reset(){
            release();
            reconnect();
            lock();
        }

    private:
        void lock();
        void release();
    };

} //namespace psrdada_cpp

#endif //PSRDADA_CPP_DADA_READ_CLIENT_HPP