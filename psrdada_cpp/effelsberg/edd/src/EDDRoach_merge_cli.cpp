#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/cli_utils.hpp"
#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/dada_input_stream.hpp"
#include "psrdada_cpp/dada_write_client.hpp"
#include "psrdada_cpp/effelsberg/edd/EDDRoach_merge.hpp"
#include "boost/program_options.hpp"


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
	key_t output_key;
	std::size_t npol;
	std::size_t nsamps_per_heap;

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
            ->default_value("dadc")
            ->notifier([&output_key](std::string out)
                {
                    output_key = string_to_key(out);
                }),
           "The shared memory key for the dada buffer to connect to (hex string)")

        ("npol,p", po::value<std::size_t>(&npol)->default_value(2),
            "Value of number of pol")

        ("nsamps_per_heap,n", po::value<std::size_t>(&nsamps_per_heap)->default_value(4096),
            "Value of samples per heap")

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
                std::cout << "EDDRoach_merge -- Read EDD data from a DADA buffer and merge the polarizations"
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
        MultiLog log("edd::EDDRoach_merge");
	DadaWriteClient output(output_key, log);
	effelsberg::edd::EDDRoach_merge mod(nsamps_per_heap, npol, output);
        DadaInputStream <decltype(mod)> input(input_key, log, mod);
        input.start();
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
