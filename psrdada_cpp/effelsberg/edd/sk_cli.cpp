#include<iostream>
#include "ReadData.hpp"
#include "SpectralKurtosis.hpp"
using namespace std;

int main(){
	ReadData r("/media/scratch/BB_data/sk_test_data/after_sk/check/2019-08-24-21:04:10_0000000000000000.000000.dada");
	r.read_file();
	printf("data len: %d\n", r.sample_size);
	SpectralKurtosis sk(r.p0, 1, 1000);
	rfi_stat stat;
	stat=sk.compute_sk();
	printf("rfi_fraction = %f\n",stat.rfi_fraction);
	return 0;
}




