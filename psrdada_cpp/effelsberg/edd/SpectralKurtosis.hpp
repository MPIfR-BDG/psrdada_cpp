#include<iostream>
#include<complex>
#include<vector>
#include<numeric>

using namespace std;

typedef std::vector<int> vInt;
typedef std::vector<float> vFloat;
typedef std::complex<float> Complex;
typedef std::vector<std::complex<float>> vComplex;

struct rfi_stat{
    vInt rfi_status;
    float rfi_fraction;
};
   
class SpectralKurtosis{
public:
    int nchannels; //number of channels
    int M; //window size
    int nwindows; //number of windows
    int sample_size; //size of input data
    vComplex data;
    //vInt rfi_status(nwindows);
    //float rfi_fraction;
    /**
     * @param           data          vector of complex data
     *                  nch           number of channels
     *                  window_size   widow size used in sk computation.
     *                                The value should be a multiple of size of data.
     */
    SpectralKurtosis(vComplex data, int nch, int window_size);
    /**
     * @brief          computes SK and returns rfi_fraction and rfi_status of each window.
     */
    rfi_stat compute_sk();
};


