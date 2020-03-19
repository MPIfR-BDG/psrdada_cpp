#include<iostream>
#include<fstream>
#include<complex>
#include<vector>

using namespace std;

typedef std::vector<float> vFloat;
typedef std::complex<float> Complex;
typedef std::vector<std::complex<float>> vComplex;

class ReadData{
	public:
		string fname;
		vFloat rp0, ip0, rp1, ip1;
		vComplex p0, p1;
		int sample_size;

		ReadData(string filename);
		void read_file();
};

