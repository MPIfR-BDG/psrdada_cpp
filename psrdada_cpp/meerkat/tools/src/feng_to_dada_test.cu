#include "psrdada_cpp/meerkat/tools/feng_to_dada.cuh"
#include "psrdada_cpp/meerkat/constants.hpp"
#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/multilog.hpp"

using namespace psrdada_cpp;

struct FengToDadaTester
{
    void init(RawBytes&){}
    bool operator()(RawBytes&){return false;}
};

int main()
{
    std::size_t const nantennas = 1;
    std::size_t const nchans = 256;
    std::size_t const ntimestamps = 800;
    std::size_t const nelements_per_timestamp =
        (nantennas * nchans * MEERKAT_FENG_NSAMPS_PER_HEAP);
    std::size_t const nelements = nelements_per_timestamp * ntimestamps;
    std::size_t const used_bytes = nelements*MEERKAT_FENG_NBYTES_PER_SAMPLE;

    std::vector<int> input_buffer(nelements);
    for (int timestamp=0; timestamp<ntimestamps; ++timestamp)
    {
        for (int ant=0; ant<nantennas; ++ant)
        {
            for (int chan=0; chan<nchans; ++chan)
            {
                for (int samp=0; samp<MEERKAT_FENG_NSAMPS_PER_HEAP; ++samp)
                {
                    int offset = timestamp*nelements_per_timestamp
                    + ant * nchans * MEERKAT_FENG_NSAMPS_PER_HEAP
                    + chan * MEERKAT_FENG_NSAMPS_PER_HEAP
                    + samp;
                    input_buffer[offset] = chan;
                }
            }
        }
    }
    FengToDadaTester tester;
    RawBytes bytes((char*)input_buffer.data(),used_bytes,used_bytes);
    meerkat::tools::FengToDada<FengToDadaTester> proc(nchans, tester);
    proc(bytes);
};