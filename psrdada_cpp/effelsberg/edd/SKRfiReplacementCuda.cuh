#ifndef PSRDADA_CPP_EFFELSBERG_EDD_SKRFIREPLACEMENTCUDA_CUH
#define PSRDADA_CPP_EFFELSBERG_EDD_SKRFIREPLACEMENTCUDA_CUH

#include "psrdada_cpp/common.hpp"
#include <thrust/device_vector.h>
#include <thrust/complex.h>

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {

class SKRfiReplacementCuda{
public:
    /**
     * @brief    constructor
     */
    SKRfiReplacementCuda();

    /**
     * @brief    destructor
     */
    ~SKRfiReplacementCuda();

    /**
     * @brief    Replaces RFI data with data generated using statistics of data from chosen number of clean_windows.
     *
     * @param(in)          rfi_status      rfi_status of input data stream
     * @param(in & out)    data            Data on which RFI has to be replaced. Returns the same but with RFI replaced.
     * @param(in)          clean_windows   number of clean windows used for computing data statistics.
     */
    void replace_rfi_data(const thrust::device_vector<int> &rfi_status,
                          thrust::device_vector<thrust::complex<float>> &data,
                          std::size_t clean_windows = 5);

private:

    thrust::device_vector<int> _rfi_window_indices;
    thrust::device_vector<int> _clean_window_indices;
    thrust::device_vector<thrust::complex <float>> _clean_data;
};
} //edd
} //effelsberg
} //psrdada_cpp

#endif //PSRDADA_CPP_EFFELSBERG_EDD_SKRFIREPLACEMENTCUDA_CUH
