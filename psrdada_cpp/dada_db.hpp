#ifndef PSRDADA_CPP_DADADB_HPP
#define PSRDADA_CPP_DADADB_HPP

#include "dada_def.h"
#include "ipcbuf.h"
#include <mutex>

namespace psrdada_cpp {

/**
 * @brief   A mock class for implementing the interface of a DADA DB (Header/Data Unit)
 *
 * @details A standard DADA HDU buffer is composed of a set of data buffers of equal size
 *          and a set of header buffers of equal size.
 *
 *          The standard use case is that a client connects to the header buffer, reads some
 *          metadata and then uses that to determine how to handle the data in the data buffers.
 *
 */
class DadaDB
{
    public:
        /**
         * @brief      Constructa new DadaDB instance
         *
         * @param[in]  _key    A hexidecimal shared memory key
         * @param[in]  _nbufs  The number of data buffers
         * @param[in]  _bufsz  The size in bytes of each data buffer
         * @param[in]  _nhdrs  The number of header buffers
         * @param[in]  _hdrsz  The size in bytes of each header buffer
         */
        DadaDB(uint64_t nbufs = DADA_DEFAULT_BLOCK_NUM,
                   uint64_t bufsz = DADA_DEFAULT_BLOCK_SIZE,
                   uint64_t nhdrs = IPCBUF_XFERS,
                   uint64_t hdrsz = DADA_DEFAULT_HEADER_SIZE);
        DadaDB(DadaDB const&) = delete;
        ~DadaDB();

        /**
         * @brief      Create the data and header blocks in shared memory
         */
        void create();

        /**
         * @brief      Destroy the allocated shared memory blocks
         */
        void destroy();

        /**
         * @brief      Return the number of data buffers
         */
        uint64_t num_data_buffers() const;

        /**
         * @brief      Return the size of each data buffer
         */
        uint64_t data_buffer_size() const;

        /**
         * @brief      Return the number of header buffers
         */
        uint64_t num_header_buffers() const;

        /**
         * @brief      Return the size of each header buffer
         */
        uint64_t header_buffer_size() const;

        /**
         * @brief      Return the hexidecimal shared memory key
         *
         * @detail     This key can be used by other processes to access
         *             the shared memory blocks.
         *
         * @note       This key is the key to the data blocks. To access
         *             the header blocks use key+1. This is handled under
         *             the hood by PSRDADA applocations and is handled by
         *             DadaReadClient.
         */
        key_t key() const;

    protected:
        void do_destroy();

    private:
        uint64_t _nbufs;
        uint64_t _bufsz;
        uint64_t _nhdrs;
        uint64_t _hdrsz;
        key_t _dada_key;
        ipcbuf_t _data_block;
        ipcbuf_t _header;
        bool _data_blocks_created;
        bool _header_blocks_created;
        std::mutex _lock;
};

} // namespace psrdada_cpp

#endif // PSRDADA_CPP_DADADB_HPP
