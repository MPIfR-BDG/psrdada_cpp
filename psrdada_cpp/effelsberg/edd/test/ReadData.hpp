#include "psrdada_cpp/common.hpp"

#include <fstream>
#include <complex>
#include <vector>

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {

# define DADA_HDR_SIZE 4096

class ReadData
{
public:
    std::vector<std::complex<float>> _pol0, _pol1;
    std::size_t _sample_size;
    //int _sample_size;
    ReadData(std::string filename);
    void read_file();
private:
    std::string _filename;
};

} //edd
} //effelsberg
} //psrdada_cpp
