#include "psrdada_cpp/transpose_to_dada.hpp"
#include "psrdada_cpp/cli_utils.hpp"
#include <iostream>


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
	
        size_t tocopy = ngroups * nsamples * ntime * nfreq * nchans;
        char *tmpindata = new char[tocopy / ngroups];
        char *tmpoutdata = new char[tocopy];

        /*for (int isamp = 0; isamp < tocopy; ++isamp) {
            tmpdata[isamp] = input_data.ptr()[isamp];
        }*/

        memcpy(tmpindata, input_data.ptr(), tocopy);

        //std::cout << tmpdata[0] << " " << tmpdata[tocopy - 1] << std::endl;

        size_t skipgroup = nchans * nsamples * ntime * nfreq * nbeams;
        size_t skipbeam = beamnum * nchans * nsamples * ntime * nfreq;
	size_t skipband = nchans * nsamples * ntime;
                
        size_t skipallchans = nchans * nfreq;
        size_t skipsamps = nsamples * skipallchans;

/*        for (int igroup = 0; igroup < ngroups; ++igroup) {

            memcpy(tmpindata, input_data.ptr() + skipbeam + igroup * skipgroup, tocopy / ngroups);

            for (int itime = 0; itime < ntime; ++itime) {
                
                for (int isamp = 0; isamp < nsamples; ++isamp) {

                    for (int iband = 0; iband < nfreq; ++iband) {
                        memcpy(tmpoutdata + iband * nchans + isamp * skipallchans + itime * skipsamps + igroup * skipgroup,
				tmpindata + iband * skipband + isamp * nchans + itime * nchans * nsamples,
				nchans * sizeof(char));
                    } // BAND LOOP
                } // SAMPLES LOOP
            } // TIME LOOP
        } // GROUP LOOP
*/

        for (int igroup = 0; igroup < ngroups; ++igroup) {

            memcpy(tmpindata, input_data.ptr() + skipbeam + igroup * skipgroup, tocopy / ngroups);

            for (int isamp = 0; isamp < nsamples; ++isamp) {
                
                for (int itime = 0; itime < ntime; ++itime) {

                    for (int iband = 0; iband < nfreq; ++iband) {
                        memcpy(tmpoutdata + iband * nchans + isamp * skipallchans + itime * skipsamps + igroup * skipgroup,
				tmpindata + iband * skipband + itime * nchans + isamp * nchans * ntime,
				nchans * sizeof(char));
                    } // BAND LOOP
                } // SAMPLES LOOP
            } // TIME LOOP
        } // GROUP LOOP

	

        /*for (n =0; n < ngroups; n++)
	{
        	for (j =0; j < nsamples; j++)
       		{
            		for (k = 0; k < ntime ; k++)
            		{

                		for (l = 0; l < nfreq ; l++)
                		{
                    			for (m=0;m < nchans ; m++)
                    			{
                        			tmpoutdata[a] = tmpindata[a]; //[m + ntime * nchans * nsamples * l + nchans * (j * ntime + k) + nsamples * nchans * ntime* nfreq * beamnum + ntime * nchans * nsamples * nfreq * nbeams * n];
                        			++a;
                    			}


                		}		


            		}

       		}
        
    	}*/

        memcpy(input_data.ptr(), tmpoutdata, tocopy);	
        delete [] tmpoutdata;
        delete [] tmpindata;
    }



} //transpose
} //psrdada_cpp
