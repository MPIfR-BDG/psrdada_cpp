#ifndef PSRDADA_CPP_DADA_WRITE_CLIENT_HPP
#define PSRDADA_CPP_DADA_WRITE_CLIENT_HPP

#include "psrdada_cpp/dada_client_base.hpp"
#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/common.hpp"

namespace psrdada_cpp {

    /**
     * @brief      Class that provides means for writing to
     *             a DADA ring buffer
     */
    class DadaWriteClient: public DadaClientBase
    {
    public:

        /**
         * @brief      A helper class for encapsulating
         *             the DADA buffer header blocks.
         */
        class HeaderStream
        {
        private:
            DadaWriteClient& _parent;
            std::unique_ptr<RawBytes> _current_block;

        public:
            /**
             * @brief      Create a new instance
             *
             * @param      parent  A reference to the parent writing client
             */
            HeaderStream(DadaWriteClient& parent);
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
             * @brief      Release the current header block.
             *
             * @detail     This will mark the block as filled, making it
             *             readable by reading client.
             */
            void release();
        };

        class DataStream
        {
        private:
            DadaWriteClient& _parent;
            std::unique_ptr<RawBytes> _current_block;
            std::size_t _block_idx;

        public:
            /**
             * @brief      Create a new instance
             *
             * @param      parent  A reference to the parent writing client
             */
            DataStream(DadaWriteClient& parent);
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
             * @detail     This will mark the block as filled, making it
             *             readable by reading client.
             */
            void release(bool eod=false);

            /**
             * @brief      Return the index of the currently open block
             */
            std::size_t block_idx() const;
        };

    public:
        /**
         * @brief      Create a new client for writing to a DADA buffer
         *
         * @param[in]  key   The hexidecimal shared memory key
         * @param      log   A MultiLog instance for logging buffer transactions
         */
        DadaWriteClient(key_t key, MultiLog& log);
        DadaWriteClient(DadaWriteClient const&) = delete;
        ~DadaWriteClient();

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

        void reset();

    private:
        void lock();
        void release();
        bool _locked;
        HeaderStream _header_stream;
        DataStream _data_stream;
    };

} //namespace psrdada_cpp

#endif //PSRDADA_CPP_DADA_WRITE_CLIENT_HPP
