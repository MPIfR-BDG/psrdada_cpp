#include "psrdada_cpp/cli_utils.hpp"
#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/file_input_stream.hpp"
#include "psrdada_cpp/dada_output_stream.hpp"
#include <memory>
#include <fstream>

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
        key_t output_key;
        std::size_t headersize;
        std::size_t nbytes;
        std::string filename;
        float streamtime;
        /** Define and parse the program options
 *         */
        namespace po = boost::program_options;
        po::options_description desc("Options");
        desc.add_options()
            ("output_key,k", po::value<std::string>()
             ->default_value("dada")
             ->notifier([&output_key](std::string in)
                 {
                 output_key = string_to_key(in);
                 }),
             "The shared memory key for the output dada buffer to connect to  (hex string)")
            ("help,h", "Print help messages")
            ("input_file,f", po::value<std::string>(&filename)->required(),
             "Input file to read")
            ("nbytes,n", po::value<std::size_t>(&nbytes)->required(),
             "Number of bytes to read in one DADA block")
            ("header_size,s", po::value<std::size_t>(&headersize)->required(),
             "size of header to read")
            ("time,t", po::value<float>(&streamtime) -> default_value(0.0f),
            "number of seconds to stream the file for");

/* Catch Error and program description */
         po::variables_map vm;
        try
        {
            po::store(po::parse_command_line(argc, argv, desc), vm);
            if ( vm.count("help")  )
            {
                std::cout << "Transpose2Dada -- read MeerKAT beamformed dada from DADA buffer, transpose per beam and write to an output DADA buffer"
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
 
        MultiLog log("outstream");

       /* Setting up the pipeline based on the type of sink*/

        DadaOutputStream outstream(output_key, log);

        FileInputStream<decltype(outstream)> input(filename, headersize, nbytes, outstream, streamtime);
         
        input.start();
        

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
