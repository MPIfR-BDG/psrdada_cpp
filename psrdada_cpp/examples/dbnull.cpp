#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/dada_read_client.hpp"

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
    try
    {
        std::size_t nbytes = 1<<17;
        key_t key;

        /** Define and parse the program options
        */
        namespace po = boost::program_options;
        po::options_description desc("Options");
        desc.add_options()
        ("help,h", "Print help messages")
        ("nbytes,n", po::value<std::size_t>(&nbytes)->default_value(1<<17),"Total number of bytes to read")
        ("key,k", po::value<std::string>()->default_value("dada")->notifier([&key](std::string in){key = string_to_key(in);}),
           "The shared memory key for the dada buffer to connect to (hex string)");

        po::variables_map vm;
        try
        {
            po::store(po::parse_command_line(argc, argv, desc), vm);
            if ( vm.count("help")  )
            {
                std::cout << "JunkDB -- read from DADA ring buffer" << std::endl
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

        MultiLog log("dbnull");
        DadaReadClient client(key,log);
        std::cout << "Opening header block" << std::endl;
        RawBytes& header = client.acquire_header_block();
        std::cout << "Header block is of size " << header.total_bytes() << " bytes ("<< header.used_bytes()
        << " bytes currently used)" << std::endl;
        std::cout << "There are a total of " << client.header_buffer_count() << " header buffers" << std::endl;
        std::cout << "Closing header block" << std::endl;
        client.release_header_block();
        std::size_t bytes_read = 0;
        while (bytes_read < nbytes)
        {
            std::cout << "Opening data block" << std::endl;
            RawBytes& block = client.acquire_data_block();
            std::cout << "Data block is of size " << block.total_bytes() << " bytes ("<< block.used_bytes()
            << " bytes currently used)" << std::endl;
            std::cout << "There are a total of " << client.data_buffer_count() << " data buffers" << std::endl;
            std::size_t bytes_to_read = std::min(nbytes-bytes_read, block.used_bytes());
            bytes_read += bytes_to_read;
            std::cout << "Read " << bytes_to_read << " bytes from data block" << std::endl;
            std::cout << "Closing data block" << std::endl;
            client.release_data_block();
        }
    }
    catch(std::exception& e)
    {
        std::cerr << "Unhandled Exception reached the top of main: "
        << e.what() << ", application will now exit" << std::endl;
        return ERROR_UNHANDLED_EXCEPTION;
    }
    return SUCCESS;

}