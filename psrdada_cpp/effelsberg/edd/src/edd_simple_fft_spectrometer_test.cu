#include "psrdada_cpp/common.h"
#include "psrdada_cpp/raw_bytes.h"
#include "psrdada_cpp/effelsberg/ebb/edd_simple_fft_spectrometer.cuh"

#include "thrust/host_vector.h"

int main()
{
    std::size_t size = 4096 * 12 * 4096 / 8
    thrust::host_vector<char> data;
    data.resize(size);
    RawBytes dada_input(data.data(), data.size());
    SimpleFFTSpectrometer spectrometer(8192, 1, 12, NULL);
    for (int ii=0; ii<100; ++ii)
    {
        spectrometer(dada_input);
    }
    return 0;
}