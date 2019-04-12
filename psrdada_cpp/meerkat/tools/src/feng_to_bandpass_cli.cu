#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/cli_utils.hpp"
#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/dada_input_stream.hpp"
#include "psrdada_cpp/dada_read_client.hpp"
#include "psrdada_cpp/simple_file_writer.hpp"
#include "psrdada_cpp/meerkat/tools/feng_to_bandpass.cuh"

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
        key_t key;
        std::size_t nantennas = 0;
        std::size_t nchannels = 0;
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
        ("key,k", po::value<std::string>()
            ->default_value("dada")
            ->notifier([&key](std::string in)
                {
                    key = string_to_key(in);
                }),
           "The shared memory key for the dada buffer to connect to (hex string)")
        ("nantennas,a", po::value<std::size_t>(&nantennas)->required(),
            "The number of antennas in the stream")
        ("nchannels,c", po::value<std::size_t>(&nchannels)->required(),
            "The number of frequency channels in the stream")
        ("outfile,o", po::value<std::string>(&filename)
            ->default_value(filename),
            "The output file to write bandpasses to")
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
                std::cout << "Feng2Bp -- read MeerKAT F-engine from DADA ring buffer and create bandpasses"
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
        MultiLog log("feng2bp");
        DadaReadClient reader(key, log);
        SimpleFileWriter outwriter(filename);
        meerkat::tools::FengToBandpass<SimpleFileWriter> feg2bp(nchannels, nantennas, outwriter);
        DadaInputStream<decltype(feg2bp)> stream(reader, feg2bp);
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