#ifndef PSRDADA_CPP_EFFELSBERG_EDD_SKRFIREPLACEMENT_HPP
#define PSRDADA_CPP_EFFELSBERG_EDD_SKRFIREPLACEMENT_HPP

#include "psrdada_cpp/common.hpp"
#include <vector>
#include <complex>
#include <algorithm>
#include <numeric>
#include <random>

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {

#define DEFAULT_NUM_CLEAN_WINDOWS 1 //number of clean windows used for computing DataStatistics 

struct DataStatistics
{
    float r_mean, r_sd, i_mean, i_sd;
};

class SKRfiReplacement{
public:
    /**
     * @brief    constructor
     *
     * @param(in)     rfi_status    rfi_status of input data
     *
     */
    SKRfiReplacement(const std::vector<int> &rfi_status);

    /**
     * @brief    destructor
     */
    ~SKRfiReplacement();

    /**
     * @brief    Replaces data in rfi_windows with replacement data (generated using statistics of data from clean_windows).
     *
     * @param(in & out)    data    Data on which RFI has to be replaced. Returns the same but with RFI replaced.   
     */
    void replace_rfi_data(std::vector<std::complex<float>> &data);

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
    void compute_data_statistics(const std::vector<std::complex<float>> &data, DataStatistics &stats);

    /**
     * @brief    Gathers data from DEFAULT_NUM_CLEAN_WINDOW number of clean windows and computes its statistics
     *
     * @param(in)     data                   actual data
     * @param(out)    ref_data_statistics    Statistics of data from clean windows
     */
    void get_clean_data_statistics(const std::vector<std::complex<float>> &data,
                                   DataStatistics &ref_data_statistics);

    /**
     * @brief    Generates replacement data using clean window data statistics
     *
     * @param(in)     stats                  data statistics
     * @param(out)    replacement_data       replacement data of size = _window_size generated using input stats.
     */
    void generate_replacement_data(const DataStatistics &stats, std::vector<std::complex<float>> &replacement_data);

    std::vector<int> _rfi_status;
    std::size_t _window_size;
    std::size_t _nwindows, _nrfi_windows, _nclean_windows;
    std::vector<int> _rfi_window_indices;
    std::vector<int> _clean_window_indices;
};

} //edd
} //effelsberg
} //psrdada_cpp

#endif //PSRDADA_CPP_EFFELSBERG_EDD_SKRFIREPLACEMENT_HPP
