#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/effelsberg/edd/eddfft.cuh"
#include "thrust/host_vector.h"

using namespace psrdada_cpp;

struct DummyHandler
{
    void init(RawBytes& header){}
    bool operator()(RawBytes& data){}
};

int main()
{
    int size = 4096 * 12 * 4096 / 8
    thrust::host_vector<char> data;
    data.resize(size);
    RawBytes dada_input(data.data(), data.size());
    DummyHandler _handler;
    effelsberg::edd::SimpleFFTSpectrometer<DummyHandler> spectrometer(8192, 1, 12, _handler);
    for (int ii=0; ii<100; ++ii)
    {
        spectrometer(dada_input);
    }
    return 0;
}