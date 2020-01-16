#include "psrdada_cpp/cli_utils.hpp"
#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/dada_client_base.hpp"
#include "psrdada_cpp/dada_input_stream.hpp"
#include "psrdada_cpp/dada_null_sink.hpp"
#include "psrdada_cpp/dada_output_stream.hpp"
#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/effelsberg/edd/VLBI.cuh"

#include <boost/program_options.hpp>
#include <boost/asio.hpp>

#include <iostream>
#include <string>
#include <chrono>
#include <thread>

using namespace psrdada_cpp;

namespace {
const size_t ERROR_IN_COMMAND_LINE = 1;
const size_t SUCCESS = 0;
const size_t ERROR_UNHANDLED_EXCEPTION = 2;
} // namespace

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {

class VDIF_Sender
{
  private:
    std::string destination_ip, source_ip;
    int port;
    double max_rate;

    uint32_t currentSecondFromReferenceEpoch;
    size_t noOfSendFrames;      // frames in last second

    boost::asio::ip::udp::socket socket;
    boost::asio::ip::udp::endpoint remote_endpoint;

  public:
  /**
     * @brief      Constructor.
     *
     * @param      destination_ip Address to send the udo packages to.
     * @param      port           Port to use.
     * @param      max_rate       Output data rate - Usefull to avoid burst
     * @param      io_service     boost::asio::io_Service instance to use for
     *                            communication.
     */
    VDIF_Sender(const std::string &source_ip, const std::string &destination_ip, int port, double max_rate, boost::asio::io_service&
        io_service): socket(io_service), source_ip(source_ip), destination_ip(destination_ip), port(port), max_rate(max_rate), currentSecondFromReferenceEpoch(0), noOfSendFrames(0)
    {

    }

  /**
   * @brief      A callback to be called on connection
   *             to a ring buffer.
   *
   * @detail     The first available header block in the
   *             in the ring buffer is provided as an argument.
   *             It is here that header parameters could be read
   *             if desired.
   *
   * @param      block  A RawBytes object wrapping a DADA header buffer
   */
  void init(RawBytes &block)
  {
    BOOST_LOG_TRIVIAL(debug) << "Preparing socket for communication from " << source_ip << " to " << destination_ip << " port: " << port;
    // drop header as not needed, only open socket.
    remote_endpoint = boost::asio::ip::udp::endpoint(boost::asio::ip::address::from_string(destination_ip), port);
    boost::asio::ip::address_v4 local_interface =
      boost::asio::ip::address_v4::from_string(source_ip);
    boost::asio::ip::multicast::outbound_interface option(local_interface);

    socket.open(boost::asio::ip::udp::v4());
    socket.set_option(option);
  };

  /**
   * @brief      A callback to be called on acqusition of a new
   *             data block.
   *
   * @param      block  A RawBytes object wrapping a DADA data buffer
   */
  bool operator()(RawBytes &block)
  {
    if (block.used_bytes() == 0)
    {
      BOOST_LOG_TRIVIAL(info) << "Received empty block, exiting.";
      return false;
    }
    boost::system::error_code err;
    VDIFHeaderView vdifHeader(reinterpret_cast<uint32_t*>(block.ptr()));

    size_t blockSize = block.used_bytes();

    BOOST_LOG_TRIVIAL(debug) << " Length of first frame: " << vdifHeader.getDataFrameLength() * 8  << " bytes";
    size_t counter = 0;
    size_t invalidFrames = 0;

    auto start = std::chrono::high_resolution_clock::now();
    for(char* frame_start = block.ptr(); frame_start < block.ptr() + blockSize; frame_start += vdifHeader.getDataFrameLength() * 8)
    {
      vdifHeader.setDataLocation(reinterpret_cast<uint32_t*>(frame_start));
      // skip invalid blocks
      if (!vdifHeader.isValid())
      {
        invalidFrames++;
        continue;
      }
      if (vdifHeader.getSecondsFromReferenceEpoch() > currentSecondFromReferenceEpoch)
      {
        BOOST_LOG_TRIVIAL(info) << " New second frome reference epoch: " << vdifHeader.getSecondsFromReferenceEpoch() << ", send " << noOfSendFrames << " in previous second.";

        BOOST_LOG_TRIVIAL(debug) <<     "  Previous second from refEpoch " << currentSecondFromReferenceEpoch << " delta = " << vdifHeader.getSecondsFromReferenceEpoch() - currentSecondFromReferenceEpoch;
        currentSecondFromReferenceEpoch = vdifHeader.getSecondsFromReferenceEpoch();
        noOfSendFrames = 0;
      }

      uint32_t frameLength = vdifHeader.getDataFrameLength() * 8; // in units of 8 bytes

      socket.send_to(boost::asio::buffer(frame_start, frameLength), remote_endpoint, 0, err);
      noOfSendFrames++;
      counter++;

      size_t processed_bytes = (frame_start - block.ptr()) + frameLength;
      auto elapsed_time = std::chrono::high_resolution_clock::now() - start;

      double current_rate = processed_bytes / (std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed_time).count() * 1E-9);
      if (current_rate > max_rate)
      {
        std::chrono::duration<double, std::nano> expected_time(processed_bytes / max_rate * 1E9);

        auto delay = expected_time  - elapsed_time;
        std::this_thread::sleep_for(delay);

        //BOOST_LOG_TRIVIAL(debug) << counter << " Set delay to " << delay.count()<< " ns. Current rate " << current_rate << ", processed_bytes: " << processed_bytes;
      }
      if (counter < 5)
        BOOST_LOG_TRIVIAL(debug) << counter << " Send - FN: " << vdifHeader.getDataFrameNumber() <<  ", Sec f. E.: " << vdifHeader.getSecondsFromReferenceEpoch() << " Get TS.: " << vdifHeader.getTimestamp();

    }
    BOOST_LOG_TRIVIAL(debug) << "Send " << counter << " frames of " << block.used_bytes() << " bytes total size. " << invalidFrames << " invalid frames in block.";
    return false;
  }
};



}
}
} // namespaces



int main(int argc, char **argv) {
  try {
    key_t input_key;

    int destination_port;
    std::string destination_ip, source_ip;
    double max_rate;

    /** Define and parse the program options
    */
    namespace po = boost::program_options;
    po::options_description desc("Options");

    desc.add_options()("help,h", "Print help messages");
    desc.add_options()(
        "input_key,i",
        po::value<std::string>()->default_value("dada")->notifier(
            [&input_key](std::string in) { input_key = string_to_key(in); }),
        "The shared memory key for the dada buffer to connect to (hex "
        "string)");
    desc.add_options()("dest_ip",
                       po::value<std::string>(&destination_ip)->required(),
                       "Destination IP");
    desc.add_options()("if_ip",
                       po::value<std::string>(&source_ip)->required(),
                       "IP of the interface to use");
    desc.add_options()("port",
                       po::value<int>()->default_value(8125)->notifier(
                           [&destination_port](int in) { destination_port = in; }),
                       "Destination PORT");

    desc.add_options()("max_rate",
                       po::value<double>()->default_value(1024*1024*5)->notifier(
                           [&max_rate](double in) { max_rate = in; }),
                       "Limit the output rate to [max_rate] byte/s");

    desc.add_options()(
        "log_level", po::value<std::string>()->default_value("info")->notifier(
                         [](std::string level) { set_log_level(level); }),
        "The logging level to use "
        "(trace, debug, info, warning, "
        "error)");

    po::variables_map vm;
    try {
      po::store(po::parse_command_line(argc, argv, desc), vm);
      if (vm.count("help")) {
        std::cout << "vdif_Send -- send vidf frames in a dada buffer to an ip via UDP."
                  << std::endl << desc << std::endl;
        return SUCCESS;
      }
      po::notify(vm);

    } catch (po::error &e) {
      std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
      std::cerr << desc << std::endl;
      return ERROR_IN_COMMAND_LINE;
    }

    MultiLog log("edd::vdif_send");
    DadaClientBase client(input_key, log);
    std::size_t buffer_bytes = client.data_buffer_size();


    boost::asio::io_service io_service;
    effelsberg::edd::VDIF_Sender vdif_sender(source_ip, destination_ip, destination_port, max_rate, io_service);

    DadaInputStream<decltype(vdif_sender)> istream(input_key, log, vdif_sender);
    istream.start();


  } catch (std::exception &e) {
    std::cerr << "Unhandled Exception reached the top of main: " << e.what()
              << ", application will now exit" << std::endl;
    return ERROR_UNHANDLED_EXCEPTION;
  }
  return SUCCESS;
}

