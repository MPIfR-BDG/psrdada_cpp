#include "psrdada_cpp/meerkat/apsuse/test/BeamCaptureControllerTester.hpp"
#include "psrdada_cpp/meerkat/apsuse/BeamCaptureController.hpp"
#include "psrdada_cpp/sigproc_file_writer.hpp"
#include "psrdada_cpp/common.hpp"
#include <boost/asio.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/optional.hpp>
#include <iostream>
#include <sstream>
#include <cstdlib>

namespace psrdada_cpp {
namespace meerkat {
namespace apsuse {
namespace test {

void send(boost::asio::local::stream_protocol::socket & socket, const std::string& message)
{
    try
    {
        const std::string msg = message;
        boost::asio::write( socket, boost::asio::buffer(message) );
    }
    catch (std::exception& e)
    {
        BOOST_LOG_TRIVIAL(error) << "Error in send";
        BOOST_LOG_TRIVIAL(error) << e.what();
    }
}

void build_start_message(Message& message, std::size_t nbeams)
{
    message.command = "start";
    message.directory = "./";
    for (std::size_t beam_idx = 0; beam_idx < nbeams; ++beam_idx)
    {
        message.beams.emplace_back();
        auto& metadata = message.beams.back();
        metadata.idx = beam_idx;
        std::stringstream name;
        name << "cfbf" << std::setw(5) << std::setfill('0') << beam_idx << std::setfill(' ');
        metadata.name = name.str();
        std::stringstream ra;
        ra << beam_idx << ":00:00.00";
        metadata.ra = ra.str();
        std::stringstream dec;
        dec << "-" << beam_idx*2 << ":00:00.00";
        metadata.dec = dec.str();
        metadata.source_name = "test_source";
    }
}

void build_stop_message(Message& message)
{
    message.command = "stop";
}

void build_json(std::stringstream& ss, Message const& message)
{
    boost::property_tree::ptree root;
    root.put("command", message.command);
    root.put("directory", message.directory);
    if (message.command == "start")
    {
        boost::property_tree::ptree all_beam_parameters;
        for (auto const& beam: message.beams)
        {
            boost::property_tree::ptree beam_parameters;
            beam_parameters.put<std::size_t>("idx", beam.idx);
            beam_parameters.put<std::string>("name", beam.name);
            beam_parameters.put<std::string>("source", beam.source_name);
            beam_parameters.put<std::string>("ra", beam.ra);
            beam_parameters.put<std::string>("dec", beam.dec);
            all_beam_parameters.push_back(std::make_pair("", beam_parameters));
        }
        root.add_child("beam_parameters", all_beam_parameters);
    }
    boost::property_tree::json_parser::write_json(ss, root);
    return;
}

void send_message(Message const& message, std::string const& socket_name)
{
    char message_buffer[1<<16];
    try
    {
        boost::asio::io_service io_service;
        boost::asio::local::stream_protocol::socket socket(io_service);
        boost::asio::local::stream_protocol::endpoint ep(socket_name);
        socket.connect(ep);
        //Send the message
        std::stringstream message_string;
        build_json(message_string, message);
        BOOST_LOG_TRIVIAL(debug) << "Sending message: " << message_string.str();
        message_string << "\r\n";
        send(socket, message_string.str());
        boost::system::error_code ec;
        socket.read_some(boost::asio::buffer(message_buffer), ec);
        socket.close();
        io_service.stop();
    }
    catch(std::exception& e)
    {
        BOOST_LOG_TRIVIAL(error) << "Error in send_json: " << e.what();
        throw std::runtime_error(e.what());
    }
    return;
}



void populate_header(FilHead& header)
{
    header.rawfile = "test.fil";
    header.source = "J0000+0000";
    header.az = 0.0;
    header.dec = 0.0;
    header.fch1 = 1400.0;
    header.foff = -0.03;
    header.ra = 0.0;
    header.rdm = 0.0;
    header.tsamp = 1.0;
    header.tstart = 58758.0; //corresponds to 2019-10-02-00:00:00
    header.za = 0.0;
    header.datatype = 1;
    header.barycentric = 0;
    header.ibeam = 1;
    header.machineid = 0;
    header.nbeams = 1;
    header.nbits = 8;
    header.nchans = 1024;
    header.nifs = 1;
    header.telescopeid = 1;
}

BeamCaptureControllerTester::BeamCaptureControllerTester()
    : ::testing::Test()
{
}

BeamCaptureControllerTester::~BeamCaptureControllerTester()
{
}

void BeamCaptureControllerTester::SetUp()
{
}

void BeamCaptureControllerTester::TearDown()
{
}



TEST_F(BeamCaptureControllerTester, do_nothing)
/* Test whether the files that are written are of the correct size */
{
    typedef std::vector<std::shared_ptr<SigprocFileWriter>> FileWritersType;

    std::size_t nwriters = 6;
    std::string socket_name = "/tmp/apsuse_test.sock";

    FileWritersType writers;

    for (std::size_t ii = 0; ii < nwriters; ++ii)
    {
        writers.emplace_back(std::make_shared<SigprocFileWriter>());
        auto& writer = *(writers.back());
        writer.directory("/tmp/");
        writer.tag("test");
        writer.max_filesize(4096);
    }

    BeamCaptureController<FileWritersType> controller(socket_name, writers);

    //Start the thread that listens on the socket
    controller.start();

    std::size_t nblocks = 5;
    std::size_t block_size = 2000;
    FilHead header;
    populate_header(header);
    char* header_ptr = new char[4096];
    RawBytes header_block(header_ptr, 4096, 4096);
    char* data_ptr = new char[block_size];
    RawBytes data_block(data_ptr, block_size, block_size);
    SigprocHeader parser;
    parser.write_header(header_ptr, header);

    for (auto& writer_ptr: writers)
    {
        writer_ptr->init(header_block);
    }

    // At this stage the writers should all be disabled
    // so the following call should have no effect other
    // than to update internal counters in the writer.
    for (std::size_t ii=0; ii<nblocks; ++ii)
    {
	for (auto& writer_ptr: writers)
	{
            writer_ptr->operator()(data_block);
        }
    }

    // Test that nothing has happened...

    Message start_message;
    build_start_message(start_message, nwriters);
    send_message(start_message, socket_name);

    // Sleep this thread for a while to allow the listner to pick up
    // the message and make the state change on the writers.
    // Use 1 second (overkill)
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    for (std::size_t ii = 0; ii < nblocks; ++ii)
    {
        for (auto& writer_ptr: writers)
        {
            writer_ptr->operator()(data_block);
        }
    }

    Message stop_message;
    build_stop_message(stop_message);
    send_message(stop_message, socket_name);


    // Again wait for the message to take effect
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // These writes should not produce any output
    for (std::size_t ii=0; ii<nblocks; ++ii)
    {
        for (auto& writer_ptr: writers)
        {
            writer_ptr->operator()(data_block);
        }
    }

    // What is the actual test here?
    controller.stop();
    delete[] header_ptr;
    delete[] data_ptr;
}



} //namespace test
} //namespace apsuse
} //namespace meerkat
} //namespace psrdada_cpp

