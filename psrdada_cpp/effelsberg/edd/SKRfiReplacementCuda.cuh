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

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {

#define DEFAULT_NUM_CLEAN_WINDOWS 1 //number of clean windows used for computing DataStatistics 

struct DataStatistics
{
    float r_mean, r_sd, i_mean, i_sd;
};

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
     * @param(in)          rfi_status    rfi_status of input data
     * @param(in & out)    data          Data on which RFI has to be replaced. Returns the same but with RFI replaced.   
     */
    void replace_rfi_data(const thrust::device_vector<int> &rfi_status,
                          thrust::device_vector<thrust::complex<float>> &data);

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
     * @brief    Computes statistics for the given input data. Here it is the data from few clean windows.
     *
     * @param(in)     data          data from default number of clean windows.
     * @param(out)    stats         DataStatistics of input data
     */
    void compute_data_statistics(const thrust::device_vector<thrust::complex<float>> &data, DataStatistics &stats);

    /**
     * @brief    Gathers data from DEFAULT_NUM_CLEAN_WINDOW number of clean windows and computes its statistics
     *
     * @param(in)     data                   actual data
     * @param(out)    ref_data_statistics    Statistics of data from clean windows
     */
    void get_clean_data_statistics(const thrust::device_vector<thrust::complex<float>> &data,
                                   DataStatistics &ref_data_statistics);

    /**
     * @brief    Generates replacement data using clean window data statistics
     *
     * @param(in)     stats                  data statistics
     * @param(out)    replacement_data       replacement data of size = _window_size generated using input stats.
     */
    void generate_replacement_data(const DataStatistics &stats, thrust::device_vector<thrust::complex<float>> &replacement_data);

    thrust::device_vector<int> _rfi_status;
    std::size_t _window_size;
    std::size_t _nwindows, _nrfi_windows, _nclean_windows;
    thrust::device_vector<int> _rfi_window_indices;
    thrust::device_vector<int> _clean_window_indices;
    thrust::device_vector<float> _d_vreal, _d_vimag;
};
} //edd
} //effelsberg
} //psrdada_cpp
