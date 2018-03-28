#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/cli_utils.hpp"
#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/dada_input_stream.hpp"
#include "psrdada_cpp/dada_output_stream.hpp"
#include "psrdada_cpp/transpose_to_dada.hpp"


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
 *         */
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
        ("output_key,o", po::value<std::vector<std::string> >()
            ->default_value("caca")
            ->notifier([&output_key](std::vector<std::string> in)
                {
                    std::uint32_t ii;
                    for (ii=0 ; ii < in.size(); ii++)
                    {
                        output_key[ii] = string_to_key(in[ii]);
                    }
                }), 

          "The shared memory key for the dada buffer to connect to based on the beams (hex string)") 
         ("nbeams,b", po::value<std::uint32_t>(&nbeams)->required(),
            "The number of beams in the stream")
         ("nchannels,c", po::value<std::uint32_t>(&nchannels)->required(),
            "The number of frequency channels per packet in the stream")
         ("nsamples,s", po::value<std::uint32_t>(&nsamples)->required(),
            "The number of time samples per heap in the stream")
         ("ntime,t", po::value<std::uint32_t>(&ntime)->required(),
            "The number of time samples per packet in the stream")
         ("nfreq,f", po::value<std::size_t>(&nfreq)->required(),
            "The number of frequency blocks in the stream");

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
 


      /* End Application Code */


       catch(std::exception& e)
       {
           std::cerr << "Unhandled Exception reached the top of main: "
           << e.what() << ", application will now exit" << std::endl;
           return ERROR_UNHANDLED_EXCEPTION;
       }
       return SUCCESS;
   }
}  
