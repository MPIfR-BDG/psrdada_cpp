#include "psrdada_cpp/common.hpp"
#include "ReadData.hpp"
#include "SpectralKurtosis.hpp"

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {

int main(){
	ReadData r("/media/scratch/BB_data/sk_test_data/after_sk/check/2019-08-24-21:04:10_0000000000000000.000000.dada");
	r.read_file();
	printf("data len: %d\n", r._sample_size);
	SpectralKurtosis sk(1, 1000);
	RFIStatistics status;
	sk.compute_sk(r._pol0, status);
	printf("rfi_fraction = %f\n",status.rfi_fraction);
	return 0;
}

}
}
}




