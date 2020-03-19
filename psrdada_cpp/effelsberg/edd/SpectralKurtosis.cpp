#include<iostream>
#include "SpectralKurtosis.hpp"

using namespace std;

SpectralKurtosis::SpectralKurtosis(vComplex x, int nch, int window_size){
    data = x;
    nchannels=nch;
    M=window_size;
    sample_size=data.size();
    nwindows=sample_size/M;
    if(sample_size%M != 0){
	throw runtime_error("window_size should be a multiple of vector size\n");
    }
}

rfi_stat SpectralKurtosis::compute_sk(){
    printf("comuting sk...\n");
    rfi_stat s;
    vFloat p1(sample_size), p2(sample_size);
	for(int i=0;i<sample_size;i++){
		p1[i]=pow(abs(data[i]),2);
		p2[i]=pow(p1[i],2);
	}

	//s1,s2
	vFloat s1(sample_size), s2(sample_size), sk(sample_size);
	s.rfi_status.resize(nwindows*nchannels);
	int r1,r2;
	for(int i=0;i<nwindows;i++){
             r1=i*M;
     	     r2=r1+M;
	     s1[i]=accumulate(p1.begin()+r1,p1.begin()+r2,0);
	     s2[i]=accumulate(p2.begin()+r1,p2.begin()+r2,0);
	     sk[i]=((M+1)/(M-1))*(((M*s2[i])/pow(s1[i],2))-1);
	     if(sk[i]>1.1 || sk[i]<0.9)
		 s.rfi_status[i]=1;
	     else
                 s.rfi_status[i]=0;
	}
	s.rfi_fraction = accumulate(s.rfi_status.begin(),s.rfi_status.end(),0.0)/nwindows;
	printf("RFI fraction: %f\n", s.rfi_fraction);
	printf("Done.\n");
	return s;
}

