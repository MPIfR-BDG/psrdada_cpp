#include "psrdada_cpp/meerkat/tuse/transpose_to_dada.hpp"
#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/cli_utils.hpp"
#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/dada_input_stream.hpp"
#include "psrdada_cpp/dada_output_stream.hpp"
#include "psrdada_cpp/psrdada_to_sigproc_header.hpp"
#include "boost/program_options.hpp"
#include <memory>
#include <fstream>
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
        key_t input_key;
        std::uint32_t nchans;
        std::uint32_t nsamples;
        std::uint32_t nfreq;
        std::uint32_t nbeams;
        std::uint32_t ngroups;
        std::string filename;
        std::fstream fkeys;
        key_t* output_keys = new key_t[nbeams];
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
        ("ngroups,g", po::value<std::uint32_t>(&ngroups)->required(),
          "Number of heap groups in one DADA block")
        ("nbeams,b", po::value<std::uint32_t>(&nbeams)->required(),
            "The number of beams in the stream")

        ("key_file,o", po::value<std::string> (&filename)->required(),
          "File containing the keys for each ouput dada buffer corresponding to each beam")

        ("nchannels,c", po::value<std::uint32_t>(&nchans)->required(),
            "The number of frequency channels per heap in the stream")
        ("nsamples,s", po::value<std::uint32_t>(&nsamples)->required(),
            "The number of time samples per heap in the stream")
        ("nfreq,f", po::value<std::uint32_t>(&nfreq)->required(),
            "The number of frequency subbands in the stream");

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

        /* Check size of the DADA buffers */

        /* Open file to parse all values to the key_t object*/
        fkeys.open(filename,std::fstream::in);
        std::uint32_t ii;
        for (ii=0; ii < nbeams; ii++)
        {
            std::string key;
            std::getline(fkeys,key);
            output_keys[ii] = string_to_key(key);
        }
        fkeys.close();
        /* Application Code */
        MultiLog log("outstream");
        /* Setting up the pipeline based on the type of sink*/
        std::vector<std::shared_ptr<DadaOutputStream>> outstreams;
        std::vector<std::shared_ptr<PsrDadaToSigprocHeader<DadaOutputStream>>> ptos;
        for (ii=0 ; ii < nbeams; ++ii)
        {
            outstreams.emplace_back(std::make_shared<DadaOutputStream>(output_keys[ii],log));
        }

        for (ii=0; ii < nbeams; ++ii)
        {
            ptos.emplace_back(std::make_shared<PsrDadaToSigprocHeader<DadaOutputStream>>(ii, *outstreams[ii]));
        }

        meerkat::tuse::TransposeToDada<PsrDadaToSigprocHeader<DadaOutputStream>> transpose(nbeams, std::move(ptos));
        transpose.set_nsamples(nsamples);
        transpose.set_nchans(nchans);
        transpose.set_nfreq(nfreq);
        transpose.set_ngroups(ngroups);
        transpose.set_nbeams(nbeams);
        MultiLog log1("instream");
        DadaInputStream<decltype(transpose)> input(input_key,log1,transpose);
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
