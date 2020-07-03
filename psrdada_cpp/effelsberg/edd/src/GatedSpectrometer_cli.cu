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


struct GatedSpectrometerInputParameters
{
    effelsberg::edd::DadaBufferLayout dadaBufferLayout;
    size_t selectedSideChannel;
    size_t speadHeapSize;
    size_t nSideChannels;
    size_t selectedBit;
    size_t fft_length;
    size_t naccumulate;
    unsigned int nbits;
    float input_level;
    float output_level;
    std::string filename;
    std::string output_type;
};



template<typename T,
         class InputType,
         class OutputType
    >
void launchSpectrometer(const GatedSpectrometerInputParameters &i)
{

    MultiLog log("DadaBufferLayout");
    std::cout << "Running with output_type: " << i.output_type << std::endl;
    if (i.output_type == "file")
    {
      SimpleFileWriter sink(i.filename);
      effelsberg::edd::GatedSpectrometer<decltype(sink), InputType, OutputType>
          spectrometer(i.dadaBufferLayout,
          i.selectedSideChannel, i.selectedBit,
          i.fft_length, i.naccumulate, i.nbits, i.input_level,
          i.output_level, sink);

      DadaInputStream<decltype(spectrometer)> istream(i.dadaBufferLayout.getInputkey(), log,
                                                      spectrometer);
      istream.start();
    }
    else if (i.output_type == "dada")
    {
      DadaOutputStream sink(string_to_key(i.filename), log);
      effelsberg::edd::GatedSpectrometer<decltype(sink), InputType, OutputType> spectrometer(i.dadaBufferLayout,
          i.selectedSideChannel, i.selectedBit,
          i.fft_length, i.naccumulate, i.nbits, i.input_level,
          i.output_level, sink);

      DadaInputStream<decltype(spectrometer)> istream(i.dadaBufferLayout.getInputkey(), log,
      spectrometer);
      istream.start();
    }
     else if (i.output_type == "profile")
    {
      NullSink sink;
      effelsberg::edd::GatedSpectrometer<decltype(sink),  InputType, OutputType> spectrometer(i.dadaBufferLayout,
          i.selectedSideChannel, i.selectedBit,
          i.fft_length, i.naccumulate, i.nbits, i.input_level,
          i.output_level, sink);

      std::vector<char> buffer(i.dadaBufferLayout.getBufferSize());
      cudaHostRegister(buffer.data(), buffer.size(), cudaHostRegisterPortable);
      RawBytes ib(buffer.data(), buffer.size(), buffer.size());
      spectrometer.init(ib);
      for (int i =0; i< 10; i++)
      {
        std::cout << "Profile Block: "<< i +1 << std::endl;
        spectrometer(ib);
      }

    }
    else
    {
      throw std::runtime_error("Unknown oputput-type");
    }
}


template<typename T> void io_eval(const GatedSpectrometerInputParameters &inputParameters, const std::string &input_polarizations, const std::string &output_format)
{
    if (input_polarizations == "Single" && output_format == "Power")
    {
        launchSpectrometer<T, effelsberg::edd::SinglePolarizationInput,
            effelsberg::edd::GatedPowerSpectrumOutput>(inputParameters);
    }
    else if (input_polarizations == "Dual" && output_format == "Power")
    {
       throw std::runtime_error("Not implemented yet.");
    }
    else if (input_polarizations == "Dual" && output_format == "Stokes")
    {
        launchSpectrometer<T, effelsberg::edd::DualPolarizationInput,
            effelsberg::edd::GatedFullStokesOutput>(inputParameters);
    }
    else
    {
       throw std::runtime_error("Not implemented yet.");
    }

}





int main(int argc, char **argv) {
  try {
    key_t input_key;

    GatedSpectrometerInputParameters ip;
    std::time_t now = std::time(NULL);
    std::tm *ptm = std::localtime(&now);
    char default_filename[32];
    std::strftime(default_filename, 32, "%Y-%m-%d-%H:%M:%S.bp", ptm);

    std::string input_polarizations = "Single";
    std::string output_format = "Power";
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
        "output_type", po::value<std::string>(&ip.output_type)->default_value("file"),
        "output type [dada, file, profile]. Default is file. Profile executes the spectrometer 10x on random data and passes the ouput to a null sink."
        );
    desc.add_options()(
        "output_bit_depth", po::value<unsigned int>(&output_bit_depth)->default_value(32),
        "output_bit_depth [8, 32]. Default is 32."
        );
    desc.add_options()(
        "output_key,o", po::value<std::string>(&ip.filename)->default_value(default_filename),
        "The key of the output bnuffer / name of the output file to write spectra "
        "to");

    desc.add_options()(
        "input_polarizations,p", po::value<std::string>(&input_polarizations)->default_value(input_polarizations),
        "Single, Dual");
    desc.add_options()(
        "output_format,f", po::value<std::string>(&output_format)->default_value(output_format),
        "Power, Stokes (requires dual poalriation input)");


    desc.add_options()("nsidechannelitems,s",
                       po::value<size_t>()->default_value(1)->notifier(
                           [&ip](size_t in) { ip.nSideChannels = in; }),
                       "Number of side channel items ( s >= 1)");
    desc.add_options()(
        "selected_sidechannel,e",
        po::value<size_t>()->default_value(0)->notifier(
            [&ip](size_t in) { ip.selectedSideChannel = in; }),
        "Side channel selected for evaluation.");
    desc.add_options()("selected_bit,B",
                       po::value<size_t>()->default_value(0)->notifier(
                           [&ip](size_t in) { ip.selectedBit = in; }),
                       "Side channel selected for evaluation.");
    desc.add_options()("speadheap_size",
                       po::value<size_t>()->default_value(4096)->notifier(
                           [&ip](size_t in) { ip.speadHeapSize = in; }),
                       "size of the spead data heaps. The number of the "
                       "heaps in the dada block depends on the number of "
                       "side channel items.");

    desc.add_options()("nbits,b", po::value<unsigned int>(&ip.nbits)->required(),
                       "The number of bits per sample in the "
                       "packetiser output (8 or 12)");
    desc.add_options()("fft_length,n", po::value<size_t>(&ip.fft_length)->required(),
                       "The length of the FFT to perform on the data");
    desc.add_options()("naccumulate,a",
                       po::value<size_t>(&ip.naccumulate)->required(),
                       "The number of samples to integrate in each channel");
    desc.add_options()("input_level",
                       po::value<float>()->default_value(100.)->notifier(
                           [&ip](float in) { ip.input_level = in; }),
                       "The input power level (standard "
                       "deviation, used for 8-bit conversion)");
    desc.add_options()("output_level",
                       po::value<float>()->default_value(100.)->notifier(
                           [&ip](float in) { ip.output_level = in; }),
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
      if (vm.count("output_type") && (!(ip.output_type == "dada" || ip.output_type == "file" || ip.output_type== "profile") ))
      {
        throw po::validation_error(po::validation_error::invalid_option_value, "output_type", ip.output_type);
      }

      if (vm.count("input_polarizations") && (!(input_polarizations == "Single" || input_polarizations == "Dual") ))
      {
        throw po::validation_error(po::validation_error::invalid_option_value, "input_polarizations", input_polarizations);
      }

      if (vm.count("output_format") && (!(output_format == "Power" || output_format == "Stokes") ))
      {
        throw po::validation_error(po::validation_error::invalid_option_value, "output_format", output_format);
      }

      if (!(ip.nSideChannels >= 1))
      {
        throw po::validation_error(po::validation_error::invalid_option_value, "Number of side channels must be 1 or larger!");
      }

    } catch (po::error &e) {
      std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
      std::cerr << desc << std::endl;
      return ERROR_IN_COMMAND_LINE;
    }

    if ((output_format ==  "Stokes") && (input_polarizations != "Dual"))
    {
        throw po::validation_error(po::validation_error::invalid_option_value, "Stokes output requires dual polarization input!");
    }

    ip.dadaBufferLayout.intitialize(input_key, ip.speadHeapSize, ip.nSideChannels);

    // ToDo: Supprot only single output depth
    if (output_bit_depth == 32)
    {

      io_eval<float>(ip, input_polarizations, output_format);
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

