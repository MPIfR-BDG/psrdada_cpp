#include "boost/program_options.hpp"
#include "psrdada_cpp/cli_utils.hpp"
#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/dada_client_base.hpp"
#include "psrdada_cpp/dada_input_stream.hpp"
#include "psrdada_cpp/dada_null_sink.hpp"
#include "psrdada_cpp/dada_output_stream.hpp"
#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/simple_file_writer.hpp"

#include "psrdada_cpp/effelsberg/edd/VLBI.cuh"

#include <ctime>
#include <iostream>
#include <time.h>


using namespace psrdada_cpp;


namespace {
const size_t ERROR_IN_COMMAND_LINE = 1;
const size_t SUCCESS = 0;
const size_t ERROR_UNHANDLED_EXCEPTION = 2;
} // namespace


int main(int argc, char **argv) {
  try {
    key_t input_key;
    unsigned int nbits;

    size_t speadHeapSize;

    std::time_t now = std::time(NULL);
    std::tm *ptm = std::localtime(&now);
    char buffer[32];
    std::strftime(buffer, 32, "%Y-%m-%d-%H:%M:%S.bp", ptm);
    std::string filename(buffer);
    std::string output_type = "file";

    int thread_id, station_id, payload_size;
    double sampleRate, digitizer_threshold;

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
    desc.add_options()(
        "output_type", po::value<std::string>(&output_type)->default_value(output_type),
        "output type [dada, file]. Default is file."
        );
    desc.add_options()(
        "output_key,o", po::value<std::string>(&filename)->default_value(filename),
        "The key of the output bnuffer / name of the output file to write spectra "
        "to");
    desc.add_options()("nbits,b", po::value<unsigned int>(&nbits)->required(),
                       "The number of bits per sample in the "
                       "packetiser output (8 or 12)");
    desc.add_options()("speadheap_size",
                       po::value<size_t>()->default_value(4096)->notifier(
                           [&speadHeapSize](size_t in) { speadHeapSize = in; }),
                       "size of the spead data heaps. The number of the "
                       "heaps in the dada block depends on the number of "
                       "side channel items.");
    desc.add_options()("thread_id",
                       po::value<int>()->default_value(0)->notifier(
                           [&thread_id](int in) { thread_id = in; }),
                       "Thread ID for the VDIF header");
    desc.add_options()("station_id",
                       po::value<int>()->default_value(0)->notifier(
                           [&station_id](int in) { station_id = in; }),
                       "Station ID for the VDIF header");
    desc.add_options()("payload_size",
                       po::value<int>()->default_value(5000)->notifier(
                           [&payload_size](int in) { payload_size = in; }),
                       "Payload size [bytes]");
    desc.add_options()("digitizer_threshold",
                       po::value<double>()->default_value(0.981599)->notifier(
                           [&digitizer_threshold](double in) { digitizer_threshold = in; }),
                       "Digitizer threshold in levels of the rms of the signal."
                       "The default v = 0.981599 is set according to "
                       "L. Kogan, Correction functions for digital correlators"
                       " with two and four quantization levels, "
                       "Radio Science 33 (1998), 1289-1296");
    desc.add_options()("sample_rate",
                       po::value<double>()->default_value(2.6E9)->notifier(
                           [&sampleRate](double in) { sampleRate = in; }),
                       "Sample rate of the input signal");
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
        std::cout << "VLBI -- Read EDD data from a DADA buffer "
                     "and convert it to 2 bit VLBI data in VDIF format"
                  << std::endl
                  << desc << std::endl;
        return SUCCESS;
      }

      po::notify(vm);
      if (vm.count("output_type") && (!(output_type == "dada" || output_type == "file") ))
      {
        throw po::validation_error(po::validation_error::invalid_option_value, "output_type", output_type);
      }

    } catch (po::error &e) {
      std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
      std::cerr << desc << std::endl;
      return ERROR_IN_COMMAND_LINE;
    }

    MultiLog log("edd::VLBI");
    DadaClientBase client(input_key, log);
    std::size_t buffer_bytes = client.data_buffer_size();

    effelsberg::edd::VDIFHeader vdifHeader;
    vdifHeader.setDataFrameLength((payload_size + effelsberg::edd::vlbiHeaderSize) / 8); // In units of 8 bytes (SIC) according to VDIF spec.
    vdifHeader.setNumberOfChannels(0);  // log_2 (channel number) 
    vdifHeader.setThreadId(thread_id);
    vdifHeader.setStationId(station_id);


    std::cout << "Running with output_type: " << output_type << std::endl;
    if (output_type == "file")
    {
      SimpleFileWriter sink(filename);
      effelsberg::edd::VLBI<decltype(sink)> vlbi(
          buffer_bytes, nbits,
          speadHeapSize, sampleRate, digitizer_threshold, vdifHeader, sink);

      DadaInputStream<decltype(vlbi)> istream(input_key, log, vlbi);
      istream.start();
    }
    else if (output_type == "dada")
    {
      DadaOutputStream sink(string_to_key(filename), log);
      effelsberg::edd::VLBI<decltype(sink)> vlbi(
          buffer_bytes, nbits,
          speadHeapSize, sampleRate, digitizer_threshold,  vdifHeader, sink);
      DadaInputStream<decltype(vlbi)> istream(input_key, log, vlbi);
      istream.start();
    }
    else
    {
      throw std::runtime_error("Unknown oputput-type");
    }


  } catch (std::exception &e) {
    std::cerr << "Unhandled Exception reached the top of main: " << e.what()
              << ", application will now exit" << std::endl;
    return ERROR_UNHANDLED_EXCEPTION;
  }
  return SUCCESS;
}

