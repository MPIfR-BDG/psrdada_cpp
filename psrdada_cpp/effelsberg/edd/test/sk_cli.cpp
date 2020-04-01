#include "psrdada_cpp/effelsberg/edd/SpectralKurtosis.hpp"
#include "psrdada_cpp/effelsberg/edd/test/SKTestVector.hpp"

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {
namespace test {

int main(){
    bool with_rfi = 1;
    int sample_size = 20000;
    int window_size = 200;
    SKTestVector tv(sample_size, window_size, with_rfi);
    std::vector<int> rfi_ind{2, 5, 7}; //RFI indices
    std::vector<std::complex<float>> samples;
    tv.generate_test_vector(rfi_ind, samples);
    
    int nch = 1;
    SpectralKurtosis sk(nch, window_size);
    RFIStatistics stat;
    printf("computing sk..\n");
    sk.compute_sk(samples, stat);
    printf("rfi_fraction = %f\n",stat.rfi_fraction);
    for(int i = 0; i < 10; i++){
	printf("RFI[%d] = %d\n", i, stat.rfi_status[i]);
    }
    return 0;
}

} //test
} //edd
} //effelsberg
} //psrdada_cpp

