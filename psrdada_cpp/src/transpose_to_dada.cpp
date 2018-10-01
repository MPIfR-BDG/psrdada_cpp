#include "psrdada_cpp/transpose_to_dada.hpp"
#include "psrdada_cpp/cli_utils.hpp"
#include <ctime>
#include <mutex>

namespace psrdada_cpp {

namespace transpose{

    /*
     * @brief This is the actual block that performs the
     * transpose. The format is based on the heap format
     * of SPEAD2 packets. This can change in time
     */  
    std::mutex MyMutex;
    void do_transpose(RawBytes& transposed_data, RawBytes& input_data,std::uint32_t nchans, std::uint32_t nsamples, std::uint32_t ntime, std::uint32_t nfreq, std::uint32_t beamnum, std::uint32_t nbeams, std::uint32_t ngroups)
    {
	std::lock_guard<std::mutex> guard(MyMutex);
        std::uint32_t j,k,l,m,n;
        std::uint32_t a =0;
	char* in_data = new char[ngroups*nbeams*nchans*nsamples*ntime*nfreq];
	char* out_data = new char[ngroups*nchans*nsamples*ntime*nfreq];
	std::memcpy(in_data,input_data.ptr(),input_data.total_bytes());
	clock_t start = clock();
	for (n =0; n < ngroups; n++)
	{
        	for (j =0; j < nsamples; j++)
       		{
            		for (k = 0; k < ntime ; k++)
            		{

                		for (l = 0; l < nfreq ; l++)
                		{
                    			for (m=0;m < nchans ; m++)
                    			{
                        			out_data[a] = in_data[m + ntime * nchans * nsamples * l + nchans * (j * ntime + k) + nsamples * nchans * ntime* nfreq * beamnum + ntime * nchans * nsamples * nfreq * nbeams * n];
                        			++a;
					}


                		}		


            		}

       		}
        
    	}
	clock_t stop = clock();
	std::cout << "Time Elapsed:" << ((double)(stop - start))/CLOCKS_PER_SEC << "\n";
	std::memcpy(transposed_data.ptr(),out_data,transposed_data.total_bytes());
    }



} //transpose
} //psrdada_cpp
