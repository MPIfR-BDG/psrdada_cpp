//#include "psrdada_cpp/multilog.hpp"
//#include "psrdada_cpp/raw_bytes.hpp"
//#include "psrdada_cpp/dada_write_client.hpp"

#include "boost/program_options.hpp"

#include <sys/types.h>
#include <iostream>
#include <string>
#include <sstream>
#include <ios>

namespace
{
  const size_t ERROR_IN_COMMAND_LINE = 1;
  const size_t SUCCESS = 0;
  const size_t ERROR_UNHANDLED_EXCEPTION = 2;
} // namespace

key_t string_to_key(std::string const& in)
{
    key_t key;
    std::stringstream converter;
    converter << std::hex << in;
    converter >> key;
    return key;
}

int main(int argc, char** argv)
{
    std::size_t nbytes;
    key_t key;

    try
    {
        /** Define and parse the program options
        */
        namespace po = boost::program_options;
        po::options_description desc("Options");
        desc.add_options()
        ("help,h", "Print help messages")
        ("nbytes,n", po::value<std::size_t>(&nbytes),"Total number of bytes to write")
        ("key,k", po::value<std::string>()->notifier([&key](std::string in){key = string_to_key(in);}),
           "The shared memory key for the dada buffer to connect to (hex string)");

        po::variables_map vm;
        try
        {
            po::store(po::parse_command_line(argc, argv, desc), vm);
            if ( vm.count("help")  )
            {
                std::cout << "Basic Command Line Parameter App" << std::endl << desc << std::endl;
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

        //app

    }
    catch(std::exception& e)
    {
        std::cerr << "Unhandled Exception reached the top of main: "
        << e.what() << ", application will now exit" << std::endl;
        return ERROR_UNHANDLED_EXCEPTION;
    }
    return SUCCESS;

}