#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/cli_utils.hpp"
#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/dada_client_base.hpp"
#include "psrdada_cpp/dada_input_stream.hpp"
#include "psrdada_cpp/dada_output_stream.hpp"
#include "psrdada_cpp/simple_file_writer.hpp"
#include "psrdada_cpp/effelsberg/edd/FftSpectrometer.cuh"
#include "psrdada_cpp/dada_null_sink.hpp"
#include "boost/program_options.hpp"
#include <time.h>
#include <ctime>

using namespace psrdada_cpp;

namespace
{
  const size_t ERROR_IN_COMMAND_LINE = 1;
  const size_t SUCCESS = 0;
  const size_t ERROR_UNHANDLED_EXCEPTION = 2;
} // namespace


int main(int argc, char** argv)
{
    try
    {
        key_t input_key;
        int fft_length;
        int nsamps_per_block;
        int naccumulate;
        int nbits;
        float input_level;
        std::time_t now = std::time(NULL);
        std::tm * ptm = std::localtime(&now);
        char buffer[32];
        std::strftime(buffer, 32, "%Y-%m-%d-%H:%M:%S.bp", ptm);
        std::string filename(buffer);

        /** Define and parse the program options
        */
        namespace po = boost::program_options;
        po::options_description desc("Options");
        desc.add_options()

        ("help,h", "Print help messages")
        ("input_key,i", po::value<std::string>()
            ->default_value("dada")
            ->notifier([&input_key](std::string in)
                {
                    input_key = string_to_key(in);
                }),
           "The shared memory key for the dada buffer to connect to (hex string)")

        ("fft_length,n", po::value<int>(&fft_length)->required(),
            "The length of the FFT to perform on the data")

        ("naccumulate,a", po::value<int>(&naccumulate)->required(),
            "The number of samples to integrate in each channel")

        ("nbits,b", po::value<int>(&nbits)->required(),
            "The number of bits per sample in the packetiser output (8 or 12)")

        ("input_level", po::value<float>(&input_level)->required(),
            "The input power level (standard deviation, used for 8-bit conversion)")

        ("offset", po::value<float>(&offset)->required(),
            "The power offset for data produced by the spectrometer (used for conversion back to 8-bit)")

        ("outfile,o", po::value<std::string>(&filename)
            ->default_value(filename),
            "The output file to write spectra to")

        ("log_level", po::value<std::string>()
            ->default_value("info")
            ->notifier([](std::string level)
                {
                    set_log_level(level);
                }),
            "The logging level to use (debug, info, warning, error)");

        po::variables_map vm;
        try
        {
            po::store(po::parse_command_line(argc, argv, desc), vm);
            if ( vm.count("help")  )
            {
                std::cout << "FftSpectrometer -- Read EDD data from a DADA buffer and perform a simple FFT spectrometer"
                << std::endl << desc << std::endl;
                return SUCCESS;
            }
            po::notify(vm);
        }
        catch(po::error& e)
        {
            std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
            std::cerr << desc << std::endl;
            return ERROR_IN_COMMAND_LINE;
        }
        /**
         * All the application code goes here
         */
        MultiLog log("edd::FftSpectrometer");
        DadaClientBase client(input_key, log);
        std::size_t buffer_bytes = client.data_buffer_size()
        SimpleFileWriter sink(filename);
        //NullSink sink;
        effelsberg::edd::FftSpectrometer<decltype(sink)> spectrometer(buffer_bytes, fft_length, naccumulate, nbits, input_level, sink);
        DadaInputStream<decltype(spectrometer)> istream(input_key, log, spectrometer);
        istream.start();
        /**
         * End of application code
         */
    }
    catch(std::exception& e)
    {
        std::cerr << "Unhandled Exception reached the top of main: "
        << e.what() << ", application will now exit" << std::endl;
        return ERROR_UNHANDLED_EXCEPTION;
    }
    return SUCCESS;

}