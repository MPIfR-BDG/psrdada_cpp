#include "psrdada_cpp/meerkat/apsuse/BeamCaptureController.hpp"
#include "psrdada_cpp/meerkat/tuse/transpose_to_dada.hpp"
#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/cli_utils.hpp"
#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/dada_input_stream.hpp"
#include "psrdada_cpp/sigproc_file_writer.hpp"
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
        std::size_t filesize;
        std::string socket_name;
        std::string directory;
        std::string log_level;
        /*
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
        ("nsamples,t", po::value<std::uint32_t>(&nsamples)->required(),
            "The number of time samples per heap in the stream")
        ("nfreq,f", po::value<std::uint32_t>(&nfreq)->required(),
            "The number of frequency blocks in the stream")
        ("size,s", po::value<std::size_t>(&filesize)->required(),
            "Size of each filterbank file to be written")
        ("socket", po::value<std::string>(&socket_name)
            ->default_value("/tmp/apsuse_capture.sock"),
            "The name of the control socket for enabling/disabling file writing")
        ("dir,d", po::value<std::string>(&directory)->required(),
            "The output directory to which files will be written")
        ("log_level", po::value<std::string>()
            ->default_value("info")
            ->notifier([](std::string level)
                {
                    set_log_level(level);
                }),
            "The logging level to use (debug, info, warning, error)");

        /* Catch Error and program description */
        po::variables_map vm;
        try
        {
            po::store(po::parse_command_line(argc, argv, desc), vm);
            if ( vm.count("help")  )
            {
                std::cout << "APSUSE capture -- read MeerKAT beamformed dada from DADA buffer, transpose per beam and write to an output files of specified size with control from a socket"
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

        /*Check the size of the file*/

        if ( filesize < nsamples * nchans * nfreq * ngroups )
        {
            throw std::runtime_error(std::string("Incorrect size of file. File size has to be greater than size of data per beam in one DADA block"));
        }

       /* Application Code */

        MultiLog log("outstream");

       /* Setting up the pipeline based on the type of sink*/
        std::uint32_t ii;
        std::vector<std::shared_ptr<SigprocFileWriter>> files;
        std::vector<std::shared_ptr<PsrDadaToSigprocHeader<SigprocFileWriter>>> ptos;
        for (ii=0; ii < nbeams; ++ii)
        {
            files.emplace_back(std::make_shared<SigprocFileWriter>());
            SigprocFileWriter& writer = (*files.back());
            writer.max_filesize(filesize);
            writer.directory(directory);
        }
        for (ii=0; ii < nbeams; ++ii)
        {
            //The beam number here is completely irrelvant
            ptos.emplace_back(std::make_shared<PsrDadaToSigprocHeader<SigprocFileWriter>>(0, *files[ii]));
        }
        meerkat::tuse::TransposeToDada<PsrDadaToSigprocHeader<SigprocFileWriter>> transpose(nbeams, ptos);
        transpose.set_nsamples(nsamples);
        transpose.set_nchans(nchans);
        transpose.set_nfreq(nfreq);
        transpose.set_ngroups(ngroups);
        transpose.set_nbeams(nbeams);
        MultiLog log1("instream");
        meerkat::apsuse::BeamCaptureController<decltype(files)> controller(socket_name, files);
        DadaInputStream<decltype(transpose)> input(input_key, log1, transpose);
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
