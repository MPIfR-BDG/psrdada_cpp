#ifndef PSRDADA_CPP_MEERKAT_FBFUSE_CONSTANTS_HPP
#define PSRDADA_CPP_MEERKAT_FBFUSE_CONSTANTS_HPP

#define FBFUSE_CB_TSCRUNCH 16
#define FBFUSE_CB_FSCRUNCH 1
#define FBFUSE_CB_NANTENNAS 32
#define FBFUSE_CB_ANTENNA_OFFSET 0
#define FBFUSE_CB_NBEAMS 1024
#define FBFUSE_IB_TSCRUNCH 16
#define FBFUSE_IB_FSCRUNCH 1
#define FBFUSE_IB_NANTENNAS 64
#define FBFUSE_IB_ANTENNA_OFFSET 0
#define FBFUSE_IB_NBEAMS 1              //
#define FBFUSE_TOTAL_NANTENNAS 64        // The total number of antennas in the input DADA buffer
#define FBFUSE_NCHANS 256               // The number of channels to be processes by this instance
#define FBFUSE_NCHANS_TOTAL 4096        // This is the F-engine channel count
#define FBFUSE_NSAMPLES_PER_HEAP 256
#define FBFUSE_NPOL 2
#define FBFUSE_CB_NTHREADS 1024
#define FBFUSE_CB_WARP_SIZE 32
#define FBFUSE_CB_NSAMPLES_PER_BLOCK (FBFUSE_CB_TSCRUNCH * FBFUSE_CB_NTHREADS/FBFUSE_CB_WARP_SIZE);

#endif //PSRDADA_CPP_MEERKAT_FBFUSE_CONSTANTS_HPP
