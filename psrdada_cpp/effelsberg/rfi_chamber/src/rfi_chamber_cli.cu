#include "psrdada_cpp/effelsberg/rfi_chamber/RSSpectrometer.cuh"
#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/dada_input_stream.hpp"
#include "psrdada_cpp/cli_utils.hpp"

#include "boost/program_options.hpp"

#include <sys/types.h>
#include <iostream>
#include <string>
#include <sstream>
#include <ios>
#include <algorithm>

using namespace psrdada_cpp;

namespace
{
  const size_t ERROR_IN_COMMAND_LINE = 1;
  const size_t SUCCESS = 0;
  const size_t ERROR_UNHANDLED_EXCEPTION = 2;
} // namespace

int main(int argc, char** argv)
{

    key_t dada_key;
    std::size_t input_nchans;
    std::size_t fft_length;
    std::size_t naccumulate;
    std::size_t nskip;
    std::string filename;

    try
    {
        /** Define and parse the program options
        */
        namespace po = boost::program_options;
        po::options_description desc("Options");
        desc.add_options()
        ("help,h", "Print help messages")
        ("key", po::value<std::string>()
            ->default_value("dada")
            ->notifier([&dada_key](std::string key)
                {
                    dada_key = string_to_key(key);
                }),
           "The shared memory key (hex string) for the dada buffer containing input data")
        ("input_nchans", po::value<std::size_t>(&input_nchans)
            ->default_value(1<<15),
            "The number of PFB channels in the input data")
        ("fft_length", po::value<std::size_t>(&fft_length)
            ->default_value(1<<12),
            "The length of FFT to perform on the data")
        ("naccumulate", po::value<std::size_t>(&naccumulate)
            ->default_value(10),
            "The number of spectra to accumulate before writing to disk")
        ("nskip", po::value<std::size_t>(&nskip)
            ->default_value(2),
            "The number of DADA blocks to skip before recording starts (this allows time for the stream to settle)")
        ("output,o", po::value<std::string>(&filename)
            ->required(),
            "The full path of the output file to which the final accumulated spectrum will be written")
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
                std::cout << "rsspectrometer -- Spectrometer for FSW IQ output" << std::endl
                << desc << std::endl;
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

        //

        /**
         * All the application code goes here
         */
        MultiLog log("rs_spectro");
        DadaClientBase client(dada_key, log);
        client.cuda_register_memory();
        effelsberg::rfi_chamber::RSSpectrometer spectrometer(
            input_nchans, fft_length, naccumulate, nskip, filename);
        DadaInputStream<decltype(spectrometer)> stream(dada_key, log, spectrometer);
        stream.start();
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
