#include "psrdada_cpp/meerkat/tuse/transpose_to_dada.hpp"
#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/cli_utils.hpp"
#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/dada_input_stream.hpp"
#include "psrdada_cpp/dada_null_sink.hpp"
#include "psrdada_cpp/psrdada_to_sigproc_header.hpp"
#include "boost/program_options.hpp"
#include <memory>
#include <fstream>
#include <dlfcn.h>
#include <unistd.h>
#include <csignal>
#include <ctime>

using namespace psrdada_cpp;

namespace
{
  const size_t ERROR_IN_COMMAND_LINE = 1;
  const size_t SUCCESS = 0;
  const size_t ERROR_UNHANDLED_EXCEPTION = 2;
} // namespace


namespace detail
{
    void SignalHandler(int signum)
    {
       exit(signum);
   }
}

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
        ("nchannels,c", po::value<std::uint32_t>(&nchans)->required(),
            "The number of frequency channels per packet in the stream")
        ("nsamples,s", po::value<std::uint32_t>(&nsamples)->required(),
            "The number of time samples per heap in the stream")
        ("nfreq,f", po::value<std::uint32_t>(&nfreq)->required(),
            "The number of frequency blocks in the stream");

        /* Catch Error and program description */
        po::variables_map vm;
        try
        {
            po::store(po::parse_command_line(argc, argv, desc), vm);
            if ( vm.count("help")  )
            {
                std::cout << "Transpose2sink -- read MeerKAT beamformed dada from DADA buffer, transpose per beam and write to a sink"
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

        std::signal(SIGINT,detail::SignalHandler);
        MultiLog log("outstream");

       /* Setting up the pipeline based on the type of sink*/
        std::vector<std::shared_ptr<NullSink>> nullsinks;
        std::uint32_t ii;
        for (ii=0; ii < nbeams; ++ii)
        {
            nullsinks.emplace_back(std::make_shared<NullSink>());
        }

        meerkat::tuse::TransposeToDada<NullSink> transpose(nbeams,std::move(nullsinks));
        transpose.set_nsamples(nsamples);
        transpose.set_nchans(nchans);
        transpose.set_nfreq(nfreq);
        transpose.set_ngroups(ngroups);
        transpose.set_nbeams(nbeams);
        PsrDadaToSigprocHeader<decltype(transpose)> ptos(transpose);
        MultiLog log1("instream");
        DadaInputStream<decltype(ptos)> input(input_key,log1,ptos);

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
