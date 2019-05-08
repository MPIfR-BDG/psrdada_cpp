#include "boost/program_options.hpp"
#include "psrdada_cpp/cli_utils.hpp"
#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/dada_input_stream.hpp"
#include "psrdada_cpp/dada_null_sink.hpp"
#include "psrdada_cpp/dada_output_stream.hpp"
#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/simple_file_writer.hpp"

#include "psrdada_cpp/effelsberg/edd/GatedSpectrometer.cuh"

#include <ctime>
#include <iostream>
#include <time.h>


using namespace psrdada_cpp;


namespace {
const size_t ERROR_IN_COMMAND_LINE = 1;
const size_t SUCCESS = 0;
const size_t ERROR_UNHANDLED_EXCEPTION = 2;
} // namespace


template<typename T>
void launchSpectrometer(const effelsberg::edd::DadaBufferLayout &dadaBufferLayout, const std::string &output_type, const std::string &filename, size_t selectedSideChannel, size_t selectedBit, size_t fft_length, size_t naccumulate, unsigned int nbits,  float input_level,  float output_level)
{

    MultiLog log("DadaBufferLayout");
    std::cout << "Running with output_type: " << output_type << std::endl;
    if (output_type == "file")
    {
      SimpleFileWriter sink(filename);
      effelsberg::edd::GatedSpectrometer<decltype(sink), T> spectrometer(dadaBufferLayout,
          selectedSideChannel, selectedBit,
          fft_length, naccumulate, nbits, input_level,
          output_level, sink);

      DadaInputStream<decltype(spectrometer)> istream(dadaBufferLayout.getInputkey(), log,
                                                      spectrometer);
      istream.start();
    }
    else if (output_type == "dada")
    {
      DadaOutputStream sink(string_to_key(filename), log);
      effelsberg::edd::GatedSpectrometer<decltype(sink), T> spectrometer(dadaBufferLayout,
          selectedSideChannel, selectedBit,
          fft_length, naccumulate, nbits, input_level,
          output_level, sink);

      DadaInputStream<decltype(spectrometer)> istream(dadaBufferLayout.getInputkey(), log,
      spectrometer);
      istream.start();
    }
    else
    {
      throw std::runtime_error("Unknown oputput-type");
    }
}



int main(int argc, char **argv) {
  try {
    key_t input_key;
    int fft_length;
    int naccumulate;
    unsigned int nbits;
    size_t nSideChannels;
    size_t selectedSideChannel;
    size_t selectedBit;
    size_t speadHeapSize;
    float input_level;
    float output_level;
    std::time_t now = std::time(NULL);
    std::tm *ptm = std::localtime(&now);
    char buffer[32];
    std::strftime(buffer, 32, "%Y-%m-%d-%H:%M:%S.bp", ptm);
    std::string filename(buffer);
    std::string output_type = "file";
    unsigned int output_bit_depth;

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
        "output_bit_depth", po::value<unsigned int>(&output_bit_depth)->default_value(8),
        "output_bit_depth [8, 32]. Default is 32."
        );
    desc.add_options()(
        "output_key,o", po::value<std::string>(&filename)->default_value(filename),
        "The key of the output bnuffer / name of the output file to write spectra "
        "to");

    desc.add_options()("nsidechannelitems,s",
                       po::value<size_t>()->default_value(1)->notifier(
                           [&nSideChannels](size_t in) { nSideChannels = in; }),
                       "Number of side channel items ( s >= 1)");
    desc.add_options()(
        "selected_sidechannel,e",
        po::value<size_t>()->default_value(0)->notifier(
            [&selectedSideChannel](size_t in) { selectedSideChannel = in; }),
        "Side channel selected for evaluation.");
    desc.add_options()("selected_bit,B",
                       po::value<size_t>()->default_value(63)->notifier(
                           [&selectedBit](size_t in) { selectedBit = in; }),
                       "Side channel selected for evaluation.");
    desc.add_options()("speadheap_size",
                       po::value<size_t>()->default_value(4096)->notifier(
                           [&speadHeapSize](size_t in) { speadHeapSize = in; }),
                       "size of the spead data heaps. The number of the "
                       "heaps in the dada block depends on the number of "
                       "side channel items.");
    desc.add_options()("nbits,b", po::value<unsigned int>(&nbits)->required(),
                       "The number of bits per sample in the "
                       "packetiser output (8 or 12)");



    desc.add_options()("fft_length,n", po::value<int>(&fft_length)->required(),
                       "The length of the FFT to perform on the data");
    desc.add_options()("naccumulate,a",
                       po::value<int>(&naccumulate)->required(),
                       "The number of samples to integrate in each channel");
    desc.add_options()("input_level",
                       po::value<float>(&input_level)->required(),
                       "The input power level (standard "
                       "deviation, used for 8-bit conversion)");
    desc.add_options()("output_level",
                       po::value<float>(&output_level)->required(),
                       "The output power level (standard "
                       "deviation, used for 8-bit "
                       "conversion)");
    desc.add_options()(
        "log_level", po::value<std::string>()->default_value("info")->notifier(
                         [](std::string level) { set_log_level(level); }),
        "The logging level to use "
        "(debug, info, warning, "
        "error)");

    po::variables_map vm;
    try {
      po::store(po::parse_command_line(argc, argv, desc), vm);
      if (vm.count("help")) {
        std::cout << "GatedSpectrometer -- Read EDD data from a DADA buffer "
                     "and split the data into two streams depending on a bit "
                     "set in the side channel data. On each stream a simple "
                     "FFT spectrometer is performed."
                  << std::endl
                  << desc << std::endl;
        return SUCCESS;
      }

      po::notify(vm);
      if (vm.count("output_type") && (!(output_type == "dada" || output_type == "file") ))
      {
        throw po::validation_error(po::validation_error::invalid_option_value, "output_type", output_type);
      }

      if (!(nSideChannels >= 1))
      {
        throw po::validation_error(po::validation_error::invalid_option_value, "Number of side channels must be 1 or larger!");
      }

    } catch (po::error &e) {
      std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
      std::cerr << desc << std::endl;
      return ERROR_IN_COMMAND_LINE;
    }

    effelsberg::edd::DadaBufferLayout bufferLayout(input_key, speadHeapSize, nSideChannels);

    if (output_bit_depth == 8)
    {
      launchSpectrometer<int8_t>(bufferLayout, output_type, filename,
          selectedSideChannel, selectedBit,
       fft_length, naccumulate, nbits, input_level, output_level);
    }
    else if (output_bit_depth == 32)
    {
      launchSpectrometer<float>(bufferLayout, output_type, filename,
          selectedSideChannel, selectedBit,
       fft_length, naccumulate, nbits, input_level, output_level);
    }
    else
    {
       throw po::validation_error(po::validation_error::invalid_option_value, "Output bit depth must be 8 or 32");
    }

  } catch (std::exception &e) {
    std::cerr << "Unhandled Exception reached the top of main: " << e.what()
              << ", application will now exit" << std::endl;
    return ERROR_UNHANDLED_EXCEPTION;
  }
  return SUCCESS;
}

