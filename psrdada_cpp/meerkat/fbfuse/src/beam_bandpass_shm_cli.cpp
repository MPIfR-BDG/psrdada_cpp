#include "psrdada_cpp/meerkat/fbfuse/BeamBandpassGenerator.hpp"
#include "psrdada_cpp/simple_shm_writer.hpp"
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
    key_t input_key;
    std::string shm_key;
    std::size_t nbeams;
    std::size_t nchans_per_subband;
    std::size_t nsubbands;
    std::size_t heap_size;
    std::size_t nbuffer_acc;
    try
    {
        /** Define and parse the program options
        */
        namespace po = boost::program_options;
        po::options_description desc("Options");
        desc.add_options()

        ("help,h", "Print help messages")

        ("input_key", po::value<std::string>()
            ->required()
            ->notifier([&](std::string key)
                {
                    input_key = string_to_key(key);
                }),
           "The shared memory key (hex string) for the dada buffer containing input data (in TBFTF order)")

        ("shm_key", po::value<std::string>(&shm_key)
            ->default_value("fbfuse_beam_bandpass"),
           "The posix shared memory key for the output")

        ("nbeams", po::value<std::size_t>(&nbeams)
            ->required(),
           "The number of beams in the buffer")

        ("nchans_per_subband", po::value<std::size_t>(&nchans_per_subband)
            ->required(),
           "The number of channels per subband in the buffer")

        ("nsubbands", po::value<std::size_t>(&nsubbands)
            ->default_value(1),
           "The number of subbands in the buffer")

        ("heap_size", po::value<std::size_t>(&heap_size)
            ->default_value(8192),
           "The heap size of heaps in the buffer")

        ("nbuffer_acc", po::value<std::size_t>(&nbuffer_acc)
            ->default_value(1),
           "The number of buffers to accumulate")

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
                std::cout << "beam_bandpass -- Generate the bandpasses of beams in TBFTF order" << std::endl
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
        /**
         * All the application code goes here
         */

        DadaReadClient client(input_key, log);
        MultiLog log("beam_bandpass");
        SimpleShmWriter shm_writer(shm_key, client.header_buffer_size(), client.data_buffer_size());
        meerkat::fbfuse::BeamBandpassGenerator<decltype(shm_writer)> bandpass_generator(
            nbeams, nchans_per_subband, nsubbands, heap_size, nbuffer_acc, shm_writer);
        DadaInputStream<decltype(bandpass_generator)> stream(input_key, log, bandpass_generator);
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
