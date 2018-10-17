#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/dada_output_stream.hpp"
#include "psrdada_cpp/dada_sync_source.hpp"
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
    try
    {
        std::size_t nbytes = 0;
        key_t key;
        std::time_t sync_epoch;
        double period;
        std::size_t ts_per_block;
        /** Define and parse the program options
        */
        namespace po = boost::program_options;
        po::options_description desc("Options");
        desc.add_options()
        ("help,h", "Print help messages")
        ("nbytes,n", po::value<std::size_t>(&nbytes)
            ->default_value(0),
            "Total number of bytes to write")
        ("key,k", po::value<std::string>()
            ->default_value("dada")
            ->notifier([&key](std::string in)
                {
                    key = string_to_key(in);
                }),
            "The shared memory key for the dada buffer to connect to (hex string)")
        ("sync_epoch,s", po::value<std::size_t>()
            ->default_value(0)
            ->notifier([&sync_epoch](std::size_t in)
                {
                    sync_epoch = static_cast<std::time_t>(in);
                }),
            "The global sync time for all producing instances")
        ("period,p", po::value<double>(&period)
            ->default_value(1.0),
            "The period (in seconds) at which dada blocks are produced")
        ("ts_per_block,t", po::value<std::size_t>(&ts_per_block)
            ->default_value(8192*128),
            "The increment in timestamp between consecutive blocks")
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
                std::cout << "SyncDB -- write 1 into a DADA ring buffer at a synchronised and fixed rate" << std::endl
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

        MultiLog log("syncdb");
        DadaOutputStream out_stream(key, log);
        sync_source<decltype(out_stream)>(
            out_stream, out_stream.client().header_buffer_size(),
            out_stream.client().data_buffer_size(), nbytes,
            sync_epoch, period, ts_per_block);

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
