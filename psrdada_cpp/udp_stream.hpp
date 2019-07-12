// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
#ifndef PSRDADA_CPP_UDP_STREAM_HPP
#define PSRDADA_CPP_UDP_STREAM_HPP


#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <stdexcept>

namespace psrdada_cpp
{


class udp_server_runtime_error : public std::runtime_error
{
	public:
    	udp_server_runtime_error(const char *w) : std::runtime_error(w) {}
};


namespace udp_stream{
	int      udprecv(char *msg, size_t max_size, int socket);
	int      timed_recv(char *msg, size_t max_size, int max_wait_ms, int socket);
			
}	

template<class HandlerType>
class UdpStream
{
    public:
        UdpStream(const std::string& addr, int port, std::size_t packetsize, HandlerType& handler);
        ~UdpStream();

        /**
         * @brief: Init function takes in a text file in PSRDADA format and passes it on as a RawBytes block
         */ 
        void start(std::string filename);
        void stop();
        bool ingest();

        int                 get_socket() const;
        int                 get_port() const;
        std::string         get_addr() const;

        void 				packetsize(std::size_t size);

    private:
        HandlerType&        _handler;
        std::size_t 	    _packetsize;
        int                 _socket;
        int                 _port;
        std::string         _addr;
        struct addrinfo *   _addrinfo;
        bool                _handler_stop_request;
        bool                _running;
        bool                _stop;
};
} // namespace psrdada_cpp
#include "psrdada_cpp/detail/udp_stream.cpp"
#endif // PSRDADA_CPP_UDP_STREAM_HPP
