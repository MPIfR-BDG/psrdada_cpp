#include "psrdada_cpp/transpose_to_dada.hpp"
#include "psrdada_cpp/cli_utils.hpp"

namespace psrdada_cpp {

namespace transpose{

    /*
     * @brief This is the actual block that performs the
     * transpose. The format is based on the heap format
     * of SPEAD2 packets. This can change in time
     */  

    void do_transpose(RawBytes& transposed_data, RawBytes& input_data,std::uint32_t nchans, std::uint32_t nsamples, std::uint32_t ntime, std::uint32_t nfreq, std::uint32_t beamnum, std::uint32_t nbeams, std::uint32_t ngroups)
    {
        std::uint32_t j,k,l,m,n;
        std::uint32_t a =0;
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
                        			transposed_data.ptr()[a] = input_data.ptr()[m + ntime * nchans * nsamples * l + nchans * (j * ntime + k) + nsamples * nchans * ntime* nfreq * beamnum + ntime * nchans * nsamples * nfreq * nbeams * n];
                        			++a;
                    			}


                		}		


            		}

       		}
        
    	}
	
    }



} //transpose
} //psrdada_cpp
