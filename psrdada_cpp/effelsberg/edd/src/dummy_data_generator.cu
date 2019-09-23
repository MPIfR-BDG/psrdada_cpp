#include "boost/program_options.hpp"
#include "psrdada_cpp/cli_utils.hpp"
#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/dada_client_base.hpp"
#include "psrdada_cpp/dada_input_stream.hpp"
#include "psrdada_cpp/dada_null_sink.hpp"
#include "psrdada_cpp/dada_output_stream.hpp"
#include "psrdada_cpp/multilog.hpp"
#include <thrust/extrema.h>
#include "psrdada_cpp/effelsberg/edd/DadaBufferLayout.hpp"
#include "psrdada_cpp/effelsberg/edd/Packer.cuh"

#include <unistd.h>
#include <iomanip>
#include <cstring>

#include <ctime>
#include <iostream>
#include <time.h>


using namespace psrdada_cpp;


namespace {
const size_t ERROR_IN_COMMAND_LINE = 1;
const size_t SUCCESS = 0;
const size_t ERROR_UNHANDLED_EXCEPTION = 2;
} // namespace


__device__ __forceinline__ uint64_t swap64(uint64_t x)
{
    uint64_t result;
    uint2 t;
    asm("mov.b64 {%0,%1},%2; \n\t"
        : "=r"(t.x), "=r"(t.y) : "l"(x));
    t.x = __byte_perm(t.x, 0, 0x0123);
    t.y = __byte_perm(t.y, 0, 0x0123);
    asm("mov.b64 %0,{%1,%2}; \n\t"
        : "=l"(result) : "r"(t.y), "r"(t.x));
    return result;
}

__global__ void toNetworkEndianess(uint64_t *s, size_t N)
{ 
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; (i < N);
       i += blockDim.x * gridDim.x) {
    s[i] = swap64(s[i]);
  }
}


int main(int argc, char **argv) {
  try {
    key_t output_key;
    unsigned int input_bit_depth;
    unsigned int delay;
    size_t nSideChannels;
    size_t nblocks;

    size_t speadHeapSize;
    std::string mode;

    /** Define and parse the program options
    */
    namespace po = boost::program_options;
    po::options_description desc("Options");

    desc.add_options()("help,h", "Print help messages");
    desc.add_options()(
        "output_key,o",
        po::value<std::string>()->default_value("dada")->notifier(
            [&output_key](std::string in) {
            output_key = string_to_key(in); }),
        "The shared memory key for the dada buffer to write to (hex "
        "string)");
    desc.add_options()("input_bit_depth,b", po::value<unsigned int>(&input_bit_depth)->required(),
                       "The number of bits per sample in the "
                       "packetiser output (8 or 12)");
//    desc.add_options()("mode,m", po::value<std::string >(&mode)->required(),
//      " Type of data to generate:\n "
//      "  gated: ");
    desc.add_options()("delay,d", po::value<unsigned int>(&delay)->required(),
                       "The delay between writing two consecutive blocks [ms].");

    desc.add_options()("nblocks,n",
                       po::value<size_t>()->default_value(0)->notifier(
                           [&nblocks](size_t in) { nblocks = in; }),
                       "Number of blocks to write in total. Default 0 means no-limit.");
    desc.add_options()("speadheap_size",
                       po::value<size_t>()->default_value(4096)->notifier(
                           [&speadHeapSize](size_t in) { speadHeapSize = in; }),
                       "size of the spead data heaps. The number of the "
                       "heaps in the dada block depends on the number of "
                       "side channel items.");

    desc.add_options()("nsidechannelitems,s",
                       po::value<size_t>()->default_value(1)->notifier(
                           [&nSideChannels](size_t in) { nSideChannels = in; }),
                       "Number of side channel items ( s >= 1)");
    desc.add_options()(
        "log_level", po::value<std::string>()->default_value("info")->notifier(
                         [](std::string level) { set_log_level(level); }),
        "The logging level to use "
        "(debug, info, warning, "
        "error)");

    po::variables_map vm;
    try {
      po::store(po::parse_command_line(argc, argv, desc), vm);
      if (vm.count("help")) {
        std::cout << "Fill dada buffer with dummy data"
                  << std::endl
                  << desc << std::endl;
        return SUCCESS;
      }

       po::notify(vm);

    } catch (po::error &e) {
      std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
      std::cerr << desc << std::endl;
      return ERROR_IN_COMMAND_LINE;
    }
    if (input_bit_depth != 8)
    {
      std::cerr << " Currently only 8 bit supported!\n";
      return ERROR_IN_COMMAND_LINE;
    }

    MultiLog log("edd::DummyDataGenerator");
    DadaOutputStream sink(output_key, log);
    char header[4096];

    std::strcpy(header, "HEADER       DADA\nHDR_VERSION  1.0\nHDR_SIZE     4096\nDADA_VERSION 1.0\nFILE_SIZE    2013265920\nNBIT           32\nNDIM         2\nNPOL         1\nNCHAN     4096\nRESOLUTION   1\nDSB 1\nSYNC_TIME    1234567890\nSAMPLE_CLOCK_START 175671842316288\n");




    RawBytes headerBlock(header, 4096, 4096);
    sink.init(headerBlock);

    effelsberg::edd::DadaBufferLayout dadaBufferLayout(output_key, speadHeapSize, nSideChannels);

    size_t n_samples = dadaBufferLayout.sizeOfData() * 8 / input_bit_depth;

    size_t nFreqs = n_samples/ 2 + 1;

    thrust::device_vector<cufftComplex> input_dummy_data_freq(nFreqs);
    thrust::device_vector<float> tmp(dadaBufferLayout.sizeOfData() * 8 / input_bit_depth);
    thrust::device_vector<uint32_t> packed_data(tmp.size() * 8 / 32);
    input_dummy_data_freq[nFreqs / 3] = make_cuComplex(50.f, 0.0f);
    input_dummy_data_freq[nFreqs / 2] = make_cuComplex(20.f, 0.0f);

    cufftHandle plan;
    cufftPlan1d(&plan, tmp.size(), CUFFT_C2R, 1);
    cufftExecC2R(plan, (cufftComplex*)thrust::raw_pointer_cast(input_dummy_data_freq.data()),(cufftReal*)thrust::raw_pointer_cast(tmp.data()));


    float min = thrust::min_element(tmp.begin(), tmp.end())[0]; 
    float max = thrust::max_element(tmp.begin(), tmp.end())[0];

    effelsberg::edd::kernels::packNbit<8><<<128, 1024>>>
      (thrust::raw_pointer_cast(tmp.data()), (uint32_t*)thrust::raw_pointer_cast(packed_data.data()), tmp.size(), min, max);

    //toNetworkEndianess<<<64, 1024>>>((uint64_t*)thrust::raw_pointer_cast(packed_data.data()), packed_data.size() /2);
    thrust::host_vector<uint32_t> output(packed_data);

    // convert from 8 bit unsigned to 8 bit signed
    uint8_t *A_unsigned = reinterpret_cast<uint8_t*>(thrust::raw_pointer_cast(output.data()));
    int8_t *A_signed = reinterpret_cast<int8_t*>(thrust::raw_pointer_cast(output.data()));
    for(int i = 0; i < output.size() * 4; i++)
    {
      int f = A_unsigned[i];
      A_signed[i] = f - 128;
    }
    size_t counter = 0;
    while(true)
    {
      counter += 1;
      RawBytes dataBlock((char*) thrust::raw_pointer_cast(output.data()), output.size() * 32 / 8, output.size() * 32 / 8 );
      sink(dataBlock);
      std::cout << "Wrote " << counter << std::endl;
      if (counter == nblocks)
        break;
      usleep(delay * 1000);
    }


  } catch (std::exception &e) {
    std::cerr << "Unhandled Exception reached the top of main: " << e.what()
              << ", application will now exit" << std::endl;
    return ERROR_UNHANDLED_EXCEPTION;
  }
  return SUCCESS;
}

