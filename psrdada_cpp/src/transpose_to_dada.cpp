#include "psrdada_cpp/transpose_to_dada.hpp"
#include "psrdada_cpp/cli_utils.hpp"
#include <ctime>
#include <mutex>
#include <iostream>


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

	//std::lock_guard<std::mutex> guard(MyMutex);
	
        size_t tocopy = ngroups * nsamples * ntime * nfreq * nchans;
        char *tmpindata = new char[tocopy / ngroups];
        char *tmpoutdata = new char[tocopy];

        size_t skipgroup = nchans * nsamples * ntime * nfreq * nbeams;
        size_t skipbeam = beamnum * nchans * nsamples * ntime * nfreq;
	size_t skipband = nchans * nsamples * ntime;
                
        size_t skipallchans = nchans * nfreq;
        size_t skipsamps = ntime * skipallchans;


        for (unsigned int igroup = 0; igroup < ngroups; ++igroup) {

            memcpy(tmpindata, input_data.ptr() + skipbeam + igroup * skipgroup, tocopy / ngroups);

            for (unsigned int isamp = 0; isamp < nsamples; ++isamp) {
                
                for (unsigned int itime = 0; itime < ntime; ++itime) {

                    for (unsigned int iband = 0; iband < nfreq; ++iband) {
                        memcpy(tmpoutdata + iband * nchans + isamp * skipsamps + itime * skipallchans + igroup * tocopy/ngroups,
				tmpindata + iband * skipband + itime * nchans + isamp * nchans * ntime,
				nchans * sizeof(char));
                   } // BAND LOOP
                } // SAMPLES LOOP
           } // TIME LOOP
       } // GROUP LOOP

	

       /* for (n =0; n < ngroups; n++)
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
                        			//tmpoutdata[a] = tmpindata[a]; //[m + ntime * nchans * nsamples * l + nchans * (j * ntime + k) + nsamples * nchans * ntime* nfreq * beamnum + ntime * nchans * nsamples * nfreq * nbeams * n];
                        			++a;
					}


                		}		


            		}

       		}
        
    	}*/

        memcpy(transposed_data.ptr(), tmpoutdata, tocopy);	
        delete [] tmpoutdata;
        delete [] tmpindata;
    }



} //transpose
} //psrdada_cpp
