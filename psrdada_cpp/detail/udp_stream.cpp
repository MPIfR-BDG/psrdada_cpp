/** @brief Initialize a UDP server object.
 *  @param[in] addr  The address we receive on.
 ** @param[in] port  The port we receive from.*/

#include "psrdada_cpp/udp_stream.hpp"
#include "psrdada_cpp/raw_bytes.hpp"

namespace psrdada_cpp
{

template<class HandlerType>
UdpStream<HandlerType>::UdpStream(const std::string& addr, int port, std::size_t packetsize, HandlerType& handler)
	:
	_handler(handler), 
	_packetsize(packetsize),
	_port(port),
    _addr(addr)
{
    char decimal_port[16];
    snprintf(decimal_port, sizeof(decimal_port), "%d", _port);
    decimal_port[sizeof(decimal_port) / sizeof(decimal_port[0]) - 1] = '\0';
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_protocol = IPPROTO_UDP;
    int r(getaddrinfo(addr.c_str(), decimal_port, &hints, &_addrinfo));
    if(r != 0 || _addrinfo == NULL)
    {
        throw udp_server_runtime_error(("invalid address or port for UDP socket: \"" + addr + ":" + decimal_port + "\"").c_str());
    }
    _socket = socket(_addrinfo->ai_family, SOCK_DGRAM | SOCK_CLOEXEC, IPPROTO_UDP);
    if(_socket == -1)
    {
        freeaddrinfo(_addrinfo);
        throw udp_server_runtime_error(("could not create UDP socket for: \"" + addr + ":" + decimal_port + "\"").c_str());
    }
    r = bind(_socket, _addrinfo->ai_addr, _addrinfo->ai_addrlen);
    if(r != 0)
    {
        freeaddrinfo(_addrinfo);
        close(_socket);
        throw udp_server_runtime_error(("could not bind UDP socket with: \"" + addr + ":" + decimal_port + "\"").c_str());
    }
}

template<class HandlerType>
void UdpStream<HandlerType>::start(std::string filename)
{

    if (_running)
    {
        throw std::runtime_error("Stream is already running");
    }

    _running = true;
    BOOST_LOG_TRIVIAL(info) << "Fetching header"; 
	std::ifstream in(filename);
	std::string contents((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
	char* header = new char[4096];
	std::copy(contents.begin(),contents.end(),header);
	RawBytes header_block(header,4096,4096,false);
	_handler.init(header_block);
	delete [] header;
    while (!_stop && !_handler_stop_request)
    {
        _handler_stop_request = ingest();
    }
    _running=false;
}

template<class HandlerType>
void UdpStream<HandlerType>::stop()
{
    _stop=true;
}

template<class HandlerType>
bool UdpStream<HandlerType>::ingest()
{
	char *o_data = new char[_packetsize];
	udp_stream::udprecv(o_data, _packetsize, _socket);
	RawBytes data_block(o_data, _packetsize, _packetsize, false);
	_handler(data_block);
	delete [] o_data;
	return false;
}

/** @brief Clean up the UDP server.
 * 
 *This function frees the address info structures and close the socket.
 **/
template<class HandlerType>
UdpStream<HandlerType>::~UdpStream()
{
    freeaddrinfo(_addrinfo);
    close(_socket);
}

/**
 * @brief set the packetsize
 */
template<class HandlerType>
void UdpStream<HandlerType>::packetsize(std::size_t size)
{
	_packetsize=size;
}

/** @brief The socket used by this UDP server.
 *
 *  This function returns the socket identifier. It can be useful if you are
 *  doing a select() on many sockets.
 *  
 *  @return The socket of this UDP server.
 **/
template<class HandlerType>
int UdpStream<HandlerType>::get_socket() const
{
    return _socket;
}

/** @brief The port used by this UDP server.
 * 
 *This function returns the port attached to the UDP server. It is a copy
 *of the port specified in the constructor.
 * 
 * @return The port of the UDP server.
 **/
template<class HandlerType>
int UdpStream<HandlerType>::get_port() const
{
    return _port;
}

/*@brief Return the address of this UDP server.
 * 
 * This function returns a verbatim copy of the address as passed to the
 * constructor of the UDP server (i.e. it does not return the canonalized
 * version of the address.)
 * *
 * * @return The address as passed to the constructor.
 * */
template<class HandlerType>
std::string UdpStream<HandlerType>::get_addr() const
{
    return _addr;
}

} // namespace psrdada_cpp
