#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/effelsberg/edd/eddfft.cuh"
#include "thrust/host_vector.h"

using namespace psrdada_cpp;

#define SIZE 100663296

struct DummyHandler
{
    void init(RawBytes& header){}
    bool operator()(RawBytes& data){return false;}
};

int main()
{
    int size = SIZE * 12 / 8;
    thrust::host_vector<char, thrust::system::cuda::experimental::pinned_allocator<char> > data;
    data.resize(size);
    RawBytes dada_input(data.data(), data.size(), data.size());
    DummyHandler _handler;
    effelsberg::edd::SimpleFFTSpectrometer<DummyHandler> spectrometer(SIZE, 8192, 1, 12, _handler);
    for (int ii=0; ii<100; ++ii)
    {
        spectrometer(dada_input);
    }
    return 0;
}
