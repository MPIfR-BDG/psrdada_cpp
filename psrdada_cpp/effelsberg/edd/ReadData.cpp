#include<iostream>
#include "ReadData.hpp"

using namespace std;

ReadData::ReadData(string filename){
	fname=filename;
}
void ReadData::read_file(){
	char *x;
	ifstream dada_file;
	int length;

	dada_file.open(fname);
	if(dada_file.is_open()){
		dada_file.seekg(-4096,dada_file.end);
		length=dada_file.tellg();

		dada_file.seekg(4096);
		x=new char[length];
		dada_file.read(x,length);
	}
	dada_file.close();

	sample_size=length/4;

	p0.resize(sample_size);
	p1.resize(sample_size);

	int offset;
	for(int i=0; i<sample_size; i++){
		offset=i*4;
		p0[i]=Complex(x[offset], x[offset+1]);
		p1[i]=Complex(x[offset+2], x[offset+3]);
	}
	printf("sample_length = %d\n", sample_size);
}

