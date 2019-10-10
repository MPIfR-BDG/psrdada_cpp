#include "psrdada_cpp/meerkat/fbfuse/BufferDump.hpp"
#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/cli_utils.hpp"
#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/dada_null_sink.hpp"
#include "boost/program_options.hpp"
#include <memory>
#include <fstream>
#include <dlfcn.h>
#include <unistd.h>
#include <csignal>
#include <ctime>

using namespace psrdada_cpp;

namespace
{
  const size_t ERROR_IN_COMMAND_LINE = 1;
  const size_t SUCCESS = 0;
  const size_t ERROR_UNHANDLED_EXCEPTION = 2;
} // namespace


namespace detail
{
    void SignalHandler(int signum)
    {
       exit(signum);
   }
}

int main(int argc, char** argv)
{
    try
    {
        key_t input_key;
        std::string socket_name;
        float max_fill_level;
        std::uint32_t nantennas;
        std::uint32_t subband_nchannels;
        std::uint32_t nchannels;
        float centre_freq;
        float bandwidth;
        /**
         * Define and parse the program options
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
            "The shared memory key for the input dada buffer to connect to  (hex string)")
        ("socket_name,s", po::value<std::string>(&socket_name)->required(),
          "full path to the UNIX socket name")
        ("max_fill_level,m", po::value<float>(&max_fill_level)->required(),
            "Maximum level to fill the DADA buffer before releasing the block")
        ("nantennas,a", po::value<std::uint32_t>(&nantennas)->required(),
            "The number of antennas used")
        ("subband_nchannels,f", po::value<std::uint32_t>(&subband_nchannels)->required(),
            "The number of channels in the subband")
        ("nchannels,n", po::value<std::uint32_t>(&nchannels)->required(),
            "Total number of channels")
        ("centre_freq,c", po::value<float>(&centre_freq)->required(),
            "Centre Frequency")
        ("bandwidth,b", po::value<float>(&bandwidth)->required(),
            "Bandwidth of one subband");

        /* Catch Error and program description */
        po::variables_map vm;
        try
        {
            po::store(po::parse_command_line(argc, argv, desc), vm);
            if ( vm.count("help")  )
            {
                std::cout << "Transpose2sink -- read MeerKAT beamformed dada from DADA buffer, transpose per beam and write to a sink"
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


       /* Application Code */

        std::signal(SIGINT,detail::SignalHandler);

       /* Setting up the pipeline based on the type of sink*/
        NullSink sink;
        MultiLog log1("instream");
        meerkat::fbfuse::BufferDump<NullSink> dumper(input_key, log1,sink, socket_name, max_fill_level, nantennas, subband_nchannels, nchannels, centre_freq, bandwidth);
        dumper.start();

      /* End Application Code */

    }
    catch(std::exception& e)
    {
        std::cerr << "Unhandled Exception reached the top of main: "
        << e.what() << ", application will now exit" << std::endl;
        return ERROR_UNHANDLED_EXCEPTION;
    }
    return SUCCESS;
}
