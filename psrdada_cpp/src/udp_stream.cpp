/** @brief Initialize a UDP server object.
 *  @param[in] addr  The address we receive on.
 ** @param[in] port  The port we receive from.*/

#include "psrdada_cpp/udp_stream.hpp"

namespace psrdada_cpp
{

namespace udp_stream
{

int recv(char *msg, size_t max_size, int socket)
{
    return ::recv(socket, msg, max_size, 0);
}

/** @brief Wait for data to come in.
 *
 * This function waits for a given amount of time for data to come in. If
 * no data comes in after max_wait_ms, the function returns with -1 and
 * errno set to EAGAIN.
 *
 * The socket is expected to be a blocking socket (the default,) although
 * it is possible to setup the socket as non-blocking if necessary for
 * some other reason.
 *
 * This function blocks for a maximum amount of time as defined by
 * max_wait_ms. It may return sooner with an error or a message.
 *
 * @param[in] msg  The buffer where the message will be saved.
 * @param[in] max_size  The size of the \p msg buffer in bytes.
 * @param[in] max_wait_ms  The maximum number of milliseconds to wait for a message.
 */

int timed_recv(char *msg, size_t max_size, int max_wait_ms, int socket)
{
    fd_set s;
    FD_ZERO(&s);
    FD_SET(socket, &s);
    struct timeval timeout;
    timeout.tv_sec = max_wait_ms / 1000;
    timeout.tv_usec = (max_wait_ms % 1000) * 1000;
    int retval = select(socket + 1, &s, &s, &s, &timeout);
    if(retval == -1)
    {
        // select() set errno accordingly
        return -1;
    }
    if(retval > 0)
    {
        // our socket has data
        return ::recv(socket, msg, max_size, 0);
    }

    // our socket has no data
    errno = EAGAIN;
    return -1;
}
}
} // namespace psrdada_cpp
