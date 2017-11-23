#include "psrdada_cpp/meerkat/tools/feng_header_inserter.cuh"
#include "psrdada_cpp/cuda_utils.hpp"

#define CLOCK_RATE 1750000000.0
#define DADA_TIMESTR "%Y-%m-%d-%H:%M:%S"

namespace psrdada_cpp {
namespace meerkat {
namespace tools {

    template <typename Handler>
    FengHeaderInserter<Handler>::FengHeaderInserter(
        Handler& handler,
        std::string const& obs_id,
        float cfreq,
        float bw,
        std::size_t nchans,
        double sync_epoch)
    : _handler(handler)
    , _obs_id(obs_id)
    , _cfreq(cfreq)
    , _bw(bw)
    , _nchans(nchans)
    , _sync_epoch(sync_epoch)
    {
    }

    template <typename Handler>
    FengHeaderInserter<Handler>::~FengHeaderInserter()
    {
    }

    template <typename Handler>
    void FengHeaderInserter<Handler>::init(RawBytes& block)
    {
        timestamp_t timestamp = *((timestamp_t*) block.ptr());
        calculate_epoch(timestamp);
        std::stringstream header;
        header
        << "HEADER       DADA\n"
        << "HDR_VERSION  1.0\n"
        << "HDR_SIZE     4096\n"
        << "DADA_VERSION 1.0\n"
        << "\n"
        << "OBS_ID       " << _obs_id << "\n"
        << "FILE_NAME    unset\n"
        << "\n"
        << "FILE_SIZE    2000000000\n"
        << "FILE_NUMBER  0\n"
        << "UTC_START    " << _utc_string << "\n"
        << "MJD_START    unset\n"
        << "\n"
        << "OBS_OFFSET   0\n"
        << "OBS_OVERLAP  0\n"
        << "\n"
        << "SAMPLE_CLOCK_START " << timestamp << "\n"
        << "SOURCE       unset\n"
        << "RA           unset\n"
        << "\n"
        << "TELESCOPE    MeerKAT\n"
        << "INSTRUMENT   SRX\n"
        << "RECEIVER     S-band\n"
        << "FREQ         " << _cfreq << "\n"
        << "BW           " << _bw << "\n"
        << "TSAMP        4.681142857142857e\n"
        << "\n"
        << "BYTES_PER_SECOND " << (_bw * 4) << "\n"
        << "NBIT         2\n"
        << "NDIM         2\n"
        << "NPOL         2\n"
        << "NCHAN        " << _nchans << "\n"
        << "RESOLUTION   1\n"
        << "DSB          1\n";
        std::memset(block.ptr(), 0, block.total_bytes());
        std::memcpy(block.ptr(), header.str().c_str(), header.str().size());
        block.used_bytes(block.total_bytes());
        _handler.init(block);
    }

    template <typename Handler>
    bool FengHeaderInserter<Handler>::operator()(RawBytes& block)
    {
        _handler(block);
        return false;
    }

    template <typename Handler>
    void FengHeaderInserter<Handler>::calculate_epoch(timestamp_t timestamp)
    {
        time_t start_utc;
        struct timeval tv;
        char tbuf[64];
        double epoch = _sync_epoch + timestamp / CLOCK_RATE;
        double integral;
        double fractional = modf(epoch, &integral);
        tv.tv_sec = integral;
        tv.tv_usec = (int) (fractional*1e6);
        struct tm *nowtm;
        start_utc = tv.tv_sec;
        nowtm = gmtime(&start_utc);
        strftime(tbuf, sizeof tbuf, DADA_TIMESTR, nowtm);
        snprintf(_utc_string, sizeof tbuf, "%s.%06ld", tbuf, tv.tv_usec);
    }


} //tools
} //meerkat
} //psrdada_cpp