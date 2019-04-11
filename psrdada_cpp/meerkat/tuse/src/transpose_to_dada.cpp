#include "psrdada_cpp/meerkat/tuse/transpose_to_dada.hpp"
#include "psrdada_cpp/cli_utils.hpp"
#include <ctime>
#include <mutex>
#include <iostream>


namespace psrdada_cpp {
namespace meerkat {
namespace tuse {

namespace transpose{

    /*
     * @brief This is the actual block that performs the
     * transpose. The format is based on the heap format
     * of SPEAD2 packets. This can change in time
     */
    std::mutex MyMutex;
    void do_transpose(RawBytes& transposed_data, RawBytes& input_data,std::uint32_t nchans, std::uint32_t nsamples, std::uint32_t nfreq, std::uint32_t beamnum, std::uint32_t nbeams, std::uint32_t ngroups)
    {
        // make copies of arrays to be transposed
        if (input_data.total_bytes() % (nfreq * nchans * nsamples * nbeams) != 0)
        {
            auto sug_size = input_data.total_bytes()/(nfreq * nchans * nsamples * nbeams);
            throw std::runtime_error(std::string("Incorrect size of the DADA block. Should be a multiple of the number of heap groups. Suggested size is:") + std::to_string(sug_size) + std::string("bytes")); 
        }
        const size_t tocopy = ngroups * nsamples * nfreq * nchans;
        std::vector<char> tmpindata(tocopy / ngroups);
        std::vector<char>tmpoutdata(tocopy);
        size_t skipgroup = nchans * nsamples * nfreq * nbeams;
        size_t skipbeam = beamnum * nchans * nsamples * nfreq;
        size_t skipband = nchans * nsamples;
        size_t skipallchans = nchans * nfreq;
        // actual transpose
        for (unsigned int igroup = 0; igroup < ngroups; ++igroup)
        {
            std::copy(input_data.ptr() + skipbeam + igroup * skipgroup, input_data.ptr() + skipbeam + igroup * skipgroup + tocopy / ngroups, tmpindata.begin());

            for (unsigned int isamp = 0; isamp < nsamples; ++isamp)
            {
                for (unsigned int iband = 0; iband < nfreq; ++iband)
                {
                    std::copy(tmpindata.begin() + iband * skipband + isamp * nchans, tmpindata.begin() + iband * skipband + isamp * nchans + nchans, tmpoutdata.begin() + iband * nchans + isamp * skipallchans + igroup * tocopy/ngroups);

                    /* Reverse the channel order */
                    std::reverse(tmpindata.begin() + iband * skipband + isamp * nchans, tmpindata.begin() + iband * skipband + isamp * nchans + nchans);

                } // BAND LOOP
            } // SAMPLES LOOP
        } // GROUP LOOP

        std::copy(tmpoutdata.begin(),tmpoutdata.end(), transposed_data.ptr());
    }
} //transpose
} //tuse
} //meerkat
} //psrdada_cpp
