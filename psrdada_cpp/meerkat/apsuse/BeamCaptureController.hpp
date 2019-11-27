#ifndef PSRDADA_CPP_MEERKAT_APSUSE_BEAMCAPTURECONTROLLER_HPP
#define PSRDADA_CPP_MEERKAT_APSUSE_BEAMCAPTURECONTROLLER_HPP

#include "psrdada_cpp/common.hpp"
#include <boost/asio.hpp>
#include <thread>

namespace psrdada_cpp {
namespace meerkat {
namespace apsuse {

struct BeamMetadata
{
    std::size_t idx;
    std::string name;
    std::string ra;
    std::string dec;
    std::string source_name;
};

struct Message
{
    std::string command;
    std::string directory;
    std::vector<BeamMetadata> beams;
};

template <typename FileWritersType>
class BeamCaptureController
{
public:
    explicit BeamCaptureController(
        std::string const& socket_name,
        FileWritersType& file_writers);
    BeamCaptureController(BeamCaptureController const&) = delete;
    ~BeamCaptureController();
    void start();
    void stop();

private:
    void setup();
    void listen();
    void get_message(Message& message);
    bool has_message() const;
    void disable_writers();
    void enable_writers();

private:
    std::string _socket_name;
    FileWritersType& _file_writers;
    std::unique_ptr<boost::asio::local::stream_protocol::socket> _socket;
    std::size_t _nbeams;
    bool _stop;
    char _msg_buffer[1<<16];
    boost::asio::io_service _io_service;
    std::unique_ptr<boost::asio::local::stream_protocol::acceptor> _acceptor;
    std::thread _listner_thread;

};

} // apsuse
} // meerkat
} // psrdada_cpp

#include "psrdada_cpp/meerkat/apsuse/detail/BeamCaptureController.cpp"

#endif //PSRDADA_CPP_MEERKAT_APSUSE_BEAMCAPTURECONTROLLER_HPP
