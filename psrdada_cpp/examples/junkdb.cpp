#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/dada_write_client.hpp"

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
        ("nbytes,n", po::value<std::size_t>(&nbytes)->default_value(1<<17),"Total number of bytes to write")
        ("key,k", po::value<std::string>()->default_value("dada")->notifier([&key](std::string in){key = string_to_key(in);}),
           "The shared memory key for the dada buffer to connect to (hex string)");

        po::variables_map vm;
        try
        {
            po::store(po::parse_command_line(argc, argv, desc), vm);
            if ( vm.count("help")  )
            {
                std::cout << "JunkDB -- write garbage into a DADA ring buffer" << std::endl
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

        MultiLog log("junkdb");
        DadaWriteClient client(key,log);
        std::cout << "Opening header block" << std::endl;
        auto& header_stream = client.header_stream();
        RawBytes& header = header_stream.next();
        std::cout << "Header block is of size " << header.total_bytes() << " bytes ("<< header.used_bytes()
        << " bytes currently used)" << std::endl;
        std::cout << "There are a total of " << client.header_buffer_count() << " header buffers" << std::endl;
        std::fill(header.ptr(),header.ptr()+header.total_bytes(),1);
        header.used_bytes(header.total_bytes());
        std::cout << "Wrote " << header.used_bytes() << " bytes to header block" << std::endl;
        std::cout << "Closing header block" << std::endl;
        header_stream.release();

        auto& data_stream = client.data_stream();

        std::size_t bytes_written = 0;
        while (bytes_written < nbytes)
        {
            std::cout << "Opening data block" << std::endl;
            RawBytes& block = data_stream.next();
            std::cout << "Data block is of size " << block.total_bytes() << " bytes ("<< block.used_bytes()
            << " bytes currently used)" << std::endl;
            std::cout << "There are a total of " << client.data_buffer_count() << " data buffers" << std::endl;
            std::size_t bytes_to_write = std::min(nbytes-bytes_written,block.total_bytes());
            std::fill(block.ptr(),block.ptr()+bytes_to_write,3);
            block.used_bytes(bytes_to_write);
            std::cout << "Wrote " << block.used_bytes() << " bytes to data block" << std::endl;
            bytes_written += block.used_bytes();
            std::cout << "Closing data block" << std::endl;
            data_stream.release(bytes_written >= nbytes);
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