#include "psrdada_cpp/meerkat/apsuse/BeamCaptureController.hpp"
#include "psrdada_cpp/sigprocheader.hpp"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include <iostream>
#include <string>
#include <sstream>
#include <iostream>
#include <thread>

using namespace boost::property_tree;

namespace psrdada_cpp {
namespace meerkat {
namespace apsuse {

template <typename FileWritersType>
BeamCaptureController<FileWritersType>::BeamCaptureController(
    std::string const& socket_name,
    FileWritersType& file_writers)
    : _socket_name(socket_name)
    , _file_writers(file_writers)
    , _socket(nullptr)
    , _nbeams(file_writers.size())
    , _stop(false)
{
    std::memset(_msg_buffer, 0, sizeof(_msg_buffer));
}


template <typename FileWritersType>
BeamCaptureController<FileWritersType>::~BeamCaptureController()
{
    if (_socket)
    {
        BOOST_LOG_TRIVIAL(debug) << "Closing message capture socket";
        (*_socket).close();
    }
}

template <typename FileWritersType>
void BeamCaptureController<FileWritersType>::setup()
{
    boost::system::error_code ec;
    ::unlink(_socket_name.c_str()); // Remove previous binding.
    boost::asio::local::stream_protocol::endpoint ep(_socket_name);
    _acceptor.reset(new boost::asio::local::stream_protocol::acceptor(_io_service, ep));
    _acceptor->non_blocking(true);
    _socket.reset(new boost::asio::local::stream_protocol::socket(_io_service));
}

template <typename FileWritersType>
void BeamCaptureController<FileWritersType>::start()
{
    BOOST_LOG_TRIVIAL(info) << "Starting BeamCaptureController "
                            << "instance (listenting on socket '"
                            << _socket_name << "')";
    _stop = false;
    setup();
    // This needs to run in a thread...
    _listner_thread = std::thread(&BeamCaptureController<FileWritersType>::listen, this);
}

template <typename FileWritersType>
void BeamCaptureController<FileWritersType>::stop()
{
    BOOST_LOG_TRIVIAL(info) << "Stopping BeamCaptureController "
                            << "instance (listenting on socket '"
                            << _socket_name << "')";
    _stop = true;
    if (_listner_thread.joinable())
    {
        _listner_thread.join();
    }
}

template <typename FileWritersType>
void BeamCaptureController<FileWritersType>::listen()
{
    BOOST_LOG_TRIVIAL(debug) << "Listening for control messages...";
    while (!_stop)
    {
        //BOOST_LOG_TRIVIAL(debug) << "Looking for messages...";
        if (has_message())
        {
            BOOST_LOG_TRIVIAL(info) << "Received control message";
            Message message;
            try
            {
                get_message(message);
            }
            catch(std::exception& e)
            {
                BOOST_LOG_TRIVIAL(error) << e.what();
                continue;
            }

            // Here we execute the requested command

            if (message.command == "stop")
            {
                disable_writers();
            }
            else if (message.command == "start")
            {
                // Assumed that receiving a start command should trigger
                // the start of a new write.
                disable_writers();

                if (message.beams.size() > _nbeams)
                {
                    BOOST_LOG_TRIVIAL(error) << "Too many beams specified in control message. "
                                             << "Expected <= " << _nbeams << " beams.";
                    continue;
                }

                SigprocHeader parser;
                for (auto const& beam: message.beams)
                {
                    if (beam.idx >= _nbeams)
                    {
                        BOOST_LOG_TRIVIAL(error) << "Beam idx " << beam.idx
                                                 << " >= number of beams, ignoring entry";
                        continue;
                    }
                    auto& header = _file_writers[beam.idx]->header();
                    header.ra = parser.hhmmss_to_double(beam.ra);
                    header.dec = parser.hhmmss_to_double(beam.dec);
                    _file_writers[beam.idx]->tag(beam.name);
                    BOOST_LOG_TRIVIAL(info) << "Enabling file writing for beam " << beam.name;
                    _file_writers[beam.idx]->enable();
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    BOOST_LOG_TRIVIAL(debug) << "Control message listening loop complete.";
}

template <typename FileWritersType>
void BeamCaptureController<FileWritersType>::get_message(Message& message)
{

    /*
     The format of the JSON to be passed should be:

     {
       "command": "start/stop",
       "beam_parameters":
          [
             {
               "idx": 0,
               "name": "cfbf00000",
               "ra": "00:00:00.00",
               "dec": "00:00:00.00"
             },
             {
               "idx": 1,
               "name": "cfbf00001",
               "ra": "01:00:00.00",
               "dec": "01:00:00.00"
             },
             {
               "idx": 2,
               "name": "cfbf00002",
               "ra": "02:00:00.00",
               "dec": "02:00:00.00"
             }
          ]
     }

     **/
    boost::system::error_code ec;
    boost::property_tree::ptree pt;
    _socket->read_some(boost::asio::buffer(_msg_buffer), ec);
    if (ec && ec != boost::asio::error::eof)
    {
        BOOST_LOG_TRIVIAL(error) << "Error on read: " << ec.message();
        throw std::runtime_error(ec.message());
    }
    _socket->close();
    std::string message_string(_msg_buffer);
    std::memset(_msg_buffer, 0, sizeof(_msg_buffer));
    std::stringstream message_stream;
    message_stream << message_string;
    BOOST_LOG_TRIVIAL(debug) << "Received string: " << message_stream.str();
    boost::property_tree::json_parser::read_json(message_stream, pt);
    message.command = pt.get<std::string>("command");
    BOOST_LOG_TRIVIAL(info) << "Recieved command: '" << message.command << "'";
    if (message.command == "start")
    {
        BOOST_LOG_TRIVIAL(info) << "Received parameters for the following beams: \n"
                                << "Index     Name    RA    Dec";
        BOOST_FOREACH(ptree::value_type& beam, pt.get_child("beam_parameters"))
        {
            BeamMetadata metadata;
            metadata.idx = beam.second.get<std::size_t>("idx");
            metadata.name = beam.second.get<std::string>("name");
            metadata.ra = beam.second.get<std::string>("ra");
            metadata.dec = beam.second.get<std::string>("dec");
            message.beams.push_back(metadata);
            BOOST_LOG_TRIVIAL(info) << metadata.idx << "\t"
                                    << metadata.name << "\t"
                                    << metadata.ra << "\t"
                                    << metadata.dec;
        }
    }
}

template <typename FileWritersType>
bool BeamCaptureController<FileWritersType>::has_message() const
{
    boost::system::error_code ec;
    boost::asio::local::stream_protocol::endpoint ep(_socket_name);
    _acceptor->accept(*_socket, ep, ec);
    if (ec && ec != boost::asio::error::try_again)
    {
        BOOST_LOG_TRIVIAL(error) << "Error on accept: " <<  ec.message();
        //throw std::runtime_error(ec.message());
    }
    auto bytes = _socket->available(ec);
    if (bytes != 0 )
    {
        return true;
    }
    return false;
}

template <typename FileWritersType>
void BeamCaptureController<FileWritersType>::disable_writers()
{
    BOOST_LOG_TRIVIAL(info) << "Disabling all output file writers";
    for (auto& writer_ptr: _file_writers)
    {
        writer_ptr->disable();
    }
}

template <typename FileWritersType>
void BeamCaptureController<FileWritersType>::enable_writers()
{
    BOOST_LOG_TRIVIAL(info) << "Enabling all output file writers";
    for (auto& writer_ptr: _file_writers)
    {
        writer_ptr->enable();
    }
}

} // apsuse
} // meerkat
} // psrdada_cpp
