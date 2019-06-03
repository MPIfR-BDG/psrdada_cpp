#include<cstdio>
#include<cstdlib>

/* A quick C++ script that calculates the DM delay and snips out the data accordingly to store
 */

namespace
{
    double dm_delay( float f1, float f2, float dm, float tsamp)
    {
        return 2*(4. 15 * 0.001 * dm * (std::powl(1/f1,2) - std::powl(1/f2, 2)))/tsamp;
    }
}

using namespace std

class DispersionCapture
{
    public:
        DispersionCapture( RawBytes& block, float DM, float tsamp );
        ~DispersionCapture();
        std::size_t get_offset(std::uint8_t timesample, std::uint8_t channel);
        std::vector<float> get_delays();
        void set_delays( float f1, float f2 );
        RayBytes& get_snippet();
        float const& dm();
        void dm(float dm);
        void ngroups(std::size_t ngroups);
        void nchans(std::size_t nchans);
        void nfreq(std::size_t nfreq);
        void nantenna(std::size_t nantenna);
        void nsamples(std::size_t nsamples)

    private:
        RawBytes _block;
        float _dm;
        float _tsamp
        std::size_t _ngroups;
        std::size_t _nchans;
        std::size_t _nfreq;
        std::size_t _nantenna;
        std::size_t _nsamples;
        std::vector<float> _delays;
        std::size_t _offset;
};


DispersionCapture::DispersionCapture( RawBytes& block, float DM )
_block(block),
_dm(DM),
_tsamp(tsamp):
{
}

DispersionCapture::~DispersionCapture()
{
}

/* Get the offset in terms of samples from the start of the DADA block
 * This assumes a specific format of the heap (FTP). One heapgroup has all subbands and all antennas.
 */

std::size_t DispersionCapture::get_offset(std::uint8_t timesample, std::uint8_t channel)
{
    if (timesample > nsamples * ngroups)
    {
        throw std::runtime_error(std::string("Time sample number asked for is larger than the maximum number of timesamples in the block"));
        exit(1);
    }
    
    std::size_t index = (std::size_t) timesample / nsamples;
    std::size_t diff = timesample - index*nsamples;

    return 2 * index * nchan * nfreq * nsamples + (2 * diff);
}

void DispersionCapture::set_delays( float f1, float f2)
{
    float chan_bw = (f2 - f1)/(_nchans*_nfreq);
    for (std::uint8_t ii=0; ii < _nchans*_nfreq; ++ii)
    {
        _delays.push_back(dm_delay( f1, f1 + (ii * chan_bw), _dm, _tsamp));
    }

    return;
}

std::vector<float> DispersionCapture::get_delays()
{
    return _delays;
}


/* The main code that runs through the whole DADA block and returns the snippet that corresponds to the candidate only */
/* Here we assume that we have start and stop time in seconds and the UTC converted to some timesample number*/


RawBytes& DispersionCapture::get_snippet( std::size_t samples )
{
    char* optr = new char[samples * _nchans * _nfreq];
    RawBytes oblock(optr,samples * _nchans * _nfreq,0,false);
    for (std::uint8_t ii=0; ii < _ngroups; ++ii)
    {
        for (std::uint8_t jj=0; jj < _nantennas; ++jj)
        {
            set_delays(f1, f2);
            for (std::uint8_t kk = 0; kk < _nchans*_nfreq; ++kk)
            {
                std::memcpy( optr , _block.ptr() + kk*nsamples*2 + _delay[kk], samples );
                optr += samples;
            }
        }
    }

    return oblock;       
/* This is the main bit. How to snip out the dispersed burst quickly ?!?!*/
}
