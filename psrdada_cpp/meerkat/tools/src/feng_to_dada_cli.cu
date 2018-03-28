#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/cli_utils.hpp"
#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/dada_input_stream.hpp"
#include "psrdada_cpp/dada_output_stream.hpp"
#include "psrdada_cpp/simple_file_writer.hpp"
#include "psrdada_cpp/meerkat/tools/feng_to_dada.cuh"

#include "boost/program_options.hpp"

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
        key_t input_key, output_key;
        std::size_t nchannels;
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
        ("output_key,o", po::value<std::string>()
            ->default_value("caca")
            ->notifier([&output_key](std::string in)
                {
                    output_key = string_to_key(in);
                }),
           "The shared memory key for the dada buffer to connect to (hex string)")
        ("nchannels,c", po::value<std::size_t>(&nchannels)->required(),
            "The number of frequency channels in the stream")
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
                std::cout << "Feng2Dada -- read MeerKAT F-engine from DADA ring buffer and convert it to TFP order DADA data"
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
        MultiLog log("feng2dada");
        DadaOutputStream ostream(output_key, log);
        meerkat::tools::FengToDada<decltype(ostream)> feng2dada(nchannels, ostream);
        DadaInputStream<decltype(feng2dada)> istream(input_key, log, feng2dada);
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
