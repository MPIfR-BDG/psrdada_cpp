#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/dada_output_stream.hpp"
#include "psrdada_cpp/dada_sync_source.hpp"
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

template<class HandlerType>
class FakeBeams
{

public:
    FakeBeams(HandlerType& handler, std::size_t size);
    ~FakeBeams();

    void init(RawBytes& block);

    bool operator()(RawBytes& block);

    void beam_num_start(std::uint8_t beam_num_start);
	
    void beam_num_end(std::uint8_t beam_num_end);

private:
    HandlerType& _handler;
    std::uint8_t _beam_num;
    std::uint8_t _beam_num_start;
    std::uint8_t _beam_num_end;
    std::size_t _size;
};

template<class HandlerType>
FakeBeams<HandlerType>::FakeBeams(HandlerType& handler, std::size_t size):
_handler(handler),
_beam_num(1),
_beam_num_start(1),
_beam_num_end(6),
_size(size)
{
}

template<class HandlerType>
FakeBeams<HandlerType>::~FakeBeams()
{
}

template<class HandlerType>
void FakeBeams<HandlerType>::init(RawBytes& block)
{
    _handler.init(block);
}

template<class HandlerType>
void FakeBeams<HandlerType>::beam_num_start(std::uint8_t beam_num_start)
{
    _beam_num_start = beam_num_start;
}

template<class HandlerType>
void FakeBeams<HandlerType>::beam_num_end(std::uint8_t beam_num_end)
{
     _beam_num_end = beam_num_end;
}

template<class HandlerType>
bool FakeBeams<HandlerType>::operator()(RawBytes& block)
{
    _beam_num = _beam_num_start;
    char* ptr = block.ptr();
    std::size_t bytes_written = 0 ;
    while (bytes_written <= block.total_bytes())
    {
      std::vector<char> beam_data(_size,_beam_num);
      std::copy(beam_data.begin(),beam_data.end(),ptr);
      ptr += _size;
      bytes_written = bytes_written +  _size;
      ++_beam_num;
      if (_beam_num > _beam_num_end)
        _beam_num = _beam_num_start;
    }
    block.used_bytes(block.total_bytes());
    _handler(block);
    return false;
}
	


int main(int argc, char** argv)
{
    try
    {
        std::size_t nbytes = 0;
        key_t key;
        std::time_t sync_epoch;
        double period;
        std::size_t ts_per_block;
	std::uint8_t beam_num_start;
	std::uint8_t beam_num_end;
	std::size_t write_size;
        /** Define and parse the program options
        */
        namespace po = boost::program_options;
        po::options_description desc("Options");
        desc.add_options()
        ("help,h", "Print help messages")
        ("nbytes,n", po::value<std::size_t>(&nbytes)
            ->default_value(0),
            "Total number of bytes to write")
        ("key,k", po::value<std::string>()
            ->default_value("dada")
            ->notifier([&key](std::string in)
                {
                    key = string_to_key(in);
                }),
            "The shared memory key for the dada buffer to connect to (hex string)")
        ("sync_epoch,s", po::value<std::size_t>()
            ->default_value(0)
            ->notifier([&sync_epoch](std::size_t in)
                {
                    sync_epoch = static_cast<std::time_t>(in);
                }),
            "The global sync time for all producing instances")
        ("period,p", po::value<double>(&period)
            ->default_value(1.0),
            "The period (in seconds) at which dada blocks are produced")
        ("ts_per_block,t", po::value<std::size_t>(&ts_per_block)
            ->default_value(8192*128),
            "The increment in timestamp between consecutive blocks")
	("beam_start,b", po::value<std::uint8_t>(&beam_num_start)
            ->default_value(1),
            "Starting beam id for the heap")
	("beam_end,e", po::value<std::uint8_t>(&beam_num_end)
            ->default_value(1),
            "Last beam id for the heap")
	("write_size, w", po::value<std::size_t>(&write_size)
	    ->default_value(0),
	    "bytes to write per cycle. Should be equal to the heap size.")
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
                std::cout << "SyncDB -- write 1 into a DADA ring buffer at a synchronised and fixed rate" << std::endl
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

        MultiLog log("syncdb");
        DadaOutputStream out_stream(key, log);
	FakeBeams<decltype(out_stream)> fakebeams(out_stream, write_size);
	fakebeams.beam_num_start(beam_num_start);
	fakebeams.beam_num_end(beam_num_end);
        sync_source<decltype(fakebeams)>(
            fakebeams, out_stream.client().header_buffer_size(),
            out_stream.client().data_buffer_size(), nbytes,
            sync_epoch, period, ts_per_block);

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
