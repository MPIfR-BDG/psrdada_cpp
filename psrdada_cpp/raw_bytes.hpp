#ifndef PSRDADA_CPP_RAW_BYTES_HPP
#define PSRDADA_CPP_RAW_BYTES_HPP

#include "psrdada_cpp/common.hpp"

namespace psrdada_cpp {

    /**
     * @brief      Class for wrapping a raw pointer to a buffer of shared memory
     *
     * @detail     This class is used to wrap pointers to shared memory
     *             returned by calls to the lower-level DADA API.
     *
     *             This class is used to wrap buffers acquired by both reading
     *             and writing clients. For writing clients, it is necessary to
     *             set the number of bytes written using the used_bytes() method
     *             after writing. This value is used when releasing the buffer.
     */
    class RawBytes
    {
    private:
        char* _ptr;
        std::size_t _total_bytes;
        std::size_t _used_bytes;

    public:
        /**
         * @brief      Create a new RawBytes instance
         *
         * @param      ptr    The pointer to the buffer to wrap
         * @param[in]  total  The total number of bytes in the buffer
         * @param[in]  used   The number of bytes currently used in the buffer
         */
        RawBytes(char* ptr, std::size_t total, std::size_t used=0);
        RawBytes(RawBytes const&) = delete;
        ~RawBytes();

        /**
         * @brief      Get the total number of bytes in the buffer
         */
        std::size_t total_bytes();

        /**
         * @brief      Get the number of currently used bytes in the buffer
         */
        std::size_t used_bytes();

        /**
         * @brief      Set the number of currently used bytes in the buffer
         *
         * @detail     For writing clients, this method should be called after
         *             all writes are complete so that the number of used_bytes
         *             can be passed to reading clients.
         */
        void used_bytes(std::size_t);

        /**
         * @brief      Get a raw pointer to the start of the buffer
         */
        char* ptr();
    };

} //namespace psrdada_cpp
#endif //PSRDADA_CPP_RAW_BYTES_HPP