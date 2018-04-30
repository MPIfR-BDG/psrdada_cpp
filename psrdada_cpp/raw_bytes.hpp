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
         * @param      ptr            The pointer to the buffer to wrap
         * @param[in]  total          The total number of bytes in the buffer
         * @param[in]  used           The number of bytes currently used in the buffer
         * @oaram[in]  device_memory  Indicates whether the memory in this buffer resides
         *                            on a GPU or not
         */
        RawBytes(char* ptr, std::size_t total, std::size_t used=0);
        RawBytes(RawBytes const&) = delete;
        ~RawBytes();

        /**
         * @brief      Get the total number of bytes in the buffer
         */
        std::size_t total_bytes() const;

        /**
         * @brief      Get the number of currently used bytes in the buffer
         */
        std::size_t used_bytes() const;

        /**
         * @brief      Set the number of currently used bytes in the buffer
         *
         * @detail     For writing clients, this method should be called after
         *             all writes are complete so that the number of used_bytes
         *             can be passed to reading clients.
         */
        void used_bytes(std::size_t);

        /**
         * @brief Return the number of unused bytes in the buffer
         */
        std::size_t remaining_bytes() const;


        /**
         * @brief      Get a raw pointer to the start of the buffer
         */
        char* ptr();
    };


    /**
     * @brief Type-safe derivative of RawBytes for handling GPU memory
     *
     * @details This class does not add anything to RawBytes but provides
     *          type saftey and opens up opportunities for overloading in
     *          operators.
     *
     */
    class RawDeviceBytes: public RawBytes
    {
    };



} //namespace psrdada_cpp
#endif //PSRDADA_CPP_RAW_BYTES_HPP