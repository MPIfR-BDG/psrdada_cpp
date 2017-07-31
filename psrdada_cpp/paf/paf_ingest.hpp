#ifndef PSRDADA_CPP_PAF_INGEST_HPP
#define PSRDADA_CPP_PAF_INGEST_HPP

namespace psrdada_cpp {
namespace paf {
namespace constants {
    static unsigned const nchans_per_packet = 7;
    static unsigned const nsamps_per_packet = 128;
    static unsigned const npols_per_packet = 2;
    static unsigned const nbytes_per_sample = 4; //16-bit real and imaginary
    static unsigned const nbytes_per_packet = (nchans_per_packet
        * nsamps_per_packet
        * npols_per_packet
        * nbytes_per_sample);
    static float const os_chan_bw_mhz = 32.0f/27.0f;
    static float const crit_chan_bw_mhz = 1.0f;
} //namespace constants


    struct PafReceiverConfig
    {
        struct PafStreamConfig
        {

        };
        std::vector<PafStreamConfig> streams;
    };

    class PafReceiver: public UdpReceiver<PafReceiver>
    {
    public:
        PafReceiver();
    };


} //namespace paf
} //namespace psrdada_cpp



#endif //PSRDADA_CPP_PAF_INGEST_HPP