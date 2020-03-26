#include "ReadData.hpp"

namespace psrdada_cpp {
namespace effelsberg {
namespace edd{

ReadData::ReadData(std::string filename)
    : _filename(filename)
{
}

void ReadData::read_file(){
    std::vector<char> x;
    std::ifstream dada_file;
    std::size_t length;
    dada_file.open(_filename);
    if(dada_file.is_open()){
        dada_file.seekg(0, dada_file.end);
	length = dada_file.tellg();
	length -= DADA_HDR_SIZE;
	dada_file.seekg(DADA_HDR_SIZE);
	x.resize(length);
	dada_file.read((char *) &x[0], length);
    }
    dada_file.close();

    _sample_size=length/4;

    _pol0.resize(_sample_size);
    _pol1.resize(_sample_size);

    std::size_t offset;
    for(int i=0; i<_sample_size; i++){
        offset = i*4;
	_pol0[i]=std::complex<float>(x[offset], x[offset+1]);
	_pol1[i]=std::complex<float>(x[offset+2], x[offset+3]);
    }
}
} //edd
} //effelsberg
} //psrdada_cpp
