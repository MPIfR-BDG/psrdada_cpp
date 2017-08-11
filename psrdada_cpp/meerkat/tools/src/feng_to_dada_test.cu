#include "psrdada_cpp/meerkat/tools/feng_to_dada.cuh"
#include "psrdada_cpp/meerkat/constants.hpp"
#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/multilog.hpp"

#define NCHANS 256
#define NTIMESTAMPS 800

using namespace psrdada_cpp;

struct FengToDadaTester
{
    void init(RawBytes&){}
    bool operator()(RawBytes& block)
    {
        int* buffer = (int*) block.ptr();

        for (int timestamp=0; timestamp<NTIMESTAMPS*MEERKAT_FENG_NSAMPS_PER_HEAP; ++timestamp)
        {
            for (int chan=0; chan<NCHANS; ++chan)
            {
                int offset = timestamp * NCHANS + chan;
                if (buffer[offset] != chan)
                {
                    std::cerr << "Failure at timestamp: " << timestamp << "  channel: " << chan << std::endl;
                }
            }
        }
        return false;
    }
};

int main()
{
    std::size_t const nelements_per_timestamp = NCHANS * MEERKAT_FENG_NSAMPS_PER_HEAP;
    std::size_t const nelements = nelements_per_timestamp * NTIMESTAMPS;
    std::size_t const used_bytes = nelements * MEERKAT_FENG_NBYTES_PER_SAMPLE * MEERKAT_FENG_NPOL_PER_HEAP;

    std::vector<int> input_buffer(nelements);

    for (int timestamp=0; timestamp<NTIMESTAMPS; ++timestamp)
    {
        for (int chan=0; chan<NCHANS; ++chan)
        {
            for (int samp=0; samp<MEERKAT_FENG_NSAMPS_PER_HEAP; ++samp)
            {
                int offset = timestamp * nelements_per_timestamp
                + chan * MEERKAT_FENG_NSAMPS_PER_HEAP
                + samp;
                input_buffer[offset] = chan;
            }
        }
    }
    FengToDadaTester tester;
    RawBytes bytes((char*)input_buffer.data(),used_bytes,used_bytes);
    meerkat::tools::FengToDada<FengToDadaTester> proc(NCHANS, tester);
    proc(bytes);
};