#ifndef PSRDADA_CPP_EFFELSBERG_EDD_SKRFIREPLACEMENTCUDA_CUH
#define PSRDADA_CPP_EFFELSBERG_EDD_SKRFIREPLACEMENTCUDA_CUH

#include "psrdada_cpp/common.hpp"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/random/linear_congruential_engine.h>
#include <nvToolsExt.h>

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
     * @brief    Replaces data in rfi_windows with replacement data (generated using statistics of data from clean_windows).
     *
     * @param(in)          rfi_status      rfi_status of input data
     * @param(in & out)    data            Data on which RFI has to be replaced. Returns the same but with RFI replaced.   
     * @param(in)          clean_windows   number of clean windows used for computing data statistics.
     */
    void replace_rfi_data(const thrust::device_vector<int> &rfi_status, 
                          thrust::device_vector<thrust::complex<float>> &data,
                          std::size_t clean_windows = 5);

private:
    /**
     * @brief    Initializes data members of the class
     */
    void init();

    /**
     * @brief    Gets indices of clean windows, _clean_window_indices    
     */
    void get_clean_window_indices();

    /**
     * @brief    Gets indices of RFI windows, _rfi_window_indices    
     */
    void get_rfi_window_indices();

    /**
     * @brief    Computes statistics of clean (rfi free) data.
     *
     */
    void compute_clean_data_statistics();

    /**
     * @brief    Gathers data from DEFAULT_NUM_CLEAN_WINDOW number of clean windows and computes its statistics
     *
     * @param(in)     data                   actual data
     */
    void get_clean_data_statistics(const thrust::device_vector<thrust::complex<float>> &data);

    thrust::device_vector<int> _rfi_status;
    std::size_t _window_size;
    std::size_t _nwindows, _nrfi_windows, _nclean_windows;
    std::size_t _nclean_windows_stat; //number of clean windows used for computing DataStatistics 
    thrust::device_vector<int> _rfi_window_indices;
    thrust::device_vector<int> _clean_window_indices;
    thrust::device_vector<thrust::complex <float>> _clean_data;
    float _ref_mean, _ref_sd;
};
} //edd
} //effelsberg
} //psrdada_cpp

#endif //PSRDADA_CPP_EFFELSBERG_EDD_SKRFIREPLACEMENTCUDA_CUH
