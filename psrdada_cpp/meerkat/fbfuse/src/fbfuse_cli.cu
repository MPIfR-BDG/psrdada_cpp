#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/dada_input_stream.hpp"
#include "psrdada_cpp/dada_null_sink.hpp"
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
        meerkat::fbfuse::PipelineConfig config;

        /** Define and parse the program options
        */
        namespace po = boost::program_options;
        po::options_description desc("Options");
        desc.add_options()
        ("help,h", "Print help messages")
        ("input_key", po::value<std::string>()
            ->required()
            ->notifier([&config](std::string key)
                {
                    config.input_dada_key(string_to_key(key));
                }),
           "The shared memory key (hex string) for the dada buffer containing input data (in TAFTP order)")
        ("cb_key", po::value<std::string>()
            ->required()
            ->notifier([&config](std::string key)
                {
                    config.cb_dada_key(string_to_key(key));
                }),
           "The shared memory key (hex string) for the output coherent beam dada buffer")
        ("ib_key", po::value<std::string>()
            ->required()
            ->notifier([&config](std::string key)
                {
                    config.ib_dada_key(string_to_key(key));
                }),
           "The shared memory key (hex string) for the output incoherent beam dada buffer")
        ("delay_key_root", po::value<std::string>()
            ->required()
            ->notifier([&config](std::string key)
                {
                    config.delay_buffer_shm(key + "_shm");
                    config.delay_buffer_mutex(key + "_mutex");
                    config.delay_buffer_sem(key + "_sem");
                }),
           "The root of the POSIX key for the delay buffer shared memory and semaphores")
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
                std::cout << "fbfuse -- The fbfuse beamformer implementations" << std::endl
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
        MultiLog log("fbfuse");
        DadaWriteClient cb_writer(config.cb_dada_key(), log);
        DadaWriteClient ib_writer(config.ib_dada_key(), log);
        // Need to setup a base client to retrive the block size
        // for the beamformer and register the host memory.
        DadaClientBase client(config.input_dada_key(), log);
        client.cuda_register_memory();
        cb_writer.cuda_register_memory();
        ib_writer.cuda_register_memory();
        meerkat::fbfuse::Pipeline pipeline(config, cb_writer, ib_writer,
            client.data_buffer_size());
        DadaInputStream<decltype(pipeline)> stream(config.input_dada_key(), log, pipeline);
        stream.start();
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