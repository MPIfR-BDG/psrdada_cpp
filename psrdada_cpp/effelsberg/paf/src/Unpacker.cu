#include "psrdada_cpp/effelsberg/paf/Unpacker.cuh"
#include "psrdada_cpp/cuda_utils.hpp"

#define MSTR_LEN      1024
#define DADA_HDRSZ    4096

#define NCHK_BEAM             48   // How many frequency chunks we will receive, we should read the number from metadata
#define NCHAN_CHK             7
#define NSAMP_DF              128
#define NPOL_SAMP             2
#define NDIM_POL              2
#define NCHAN_IN              (NCHK_BEAM * NCHAN_CHK)

#define NBYTE_RT              8
#define NBYTE_IN              2   // 16 bits
#define NBYTE_OUT             1   // 8 bits

#define OSAMP_RATEI           0.84375  // 27.0/32.0
#define CUFFT_RANK1           1
#define CUFFT_RANK2           1         

#define CUFFT_NX1             64
#define CUFFT_MOD1            27              // Set to remove oversampled data
#define NCHAN_KEEP_CHAN       (int)(CUFFT_NX1 * OSAMP_RATEI)
#define CUFFT_NX2             (int)(CUFFT_NX1 * OSAMP_RATEI)              // We work in seperate raw channels
#define CUFFT_MOD2            (int)(CUFFT_NX2/2)         

#define NCHAN_OUT             324             // Final number of channels, multiple times of CUFFT2_NX2
#define NCHAN_KEEP_BAND       (int)(CUFFT_NX2 * NCHAN_OUT)
#define NCHAN_RATEI           (NCHAN_IN * NCHAN_KEEP_CHAN / (double)NCHAN_KEEP_BAND)

#define NCHAN_EDGE            (int)((NCHAN_IN * NCHAN_KEEP_CHAN - NCHAN_KEEP_BAND)/2)
#define TILE_DIM              CUFFT_NX2
#define NROWBLOCK_TRANS       18               // Multiple times of TILE_DIM (CUFFT_NX2)

#define SCL_DTSZ              (OSAMP_RATEI * (double)NBYTE_OUT/ (NCHAN_RATEI * (double)NBYTE_IN))
#define SCL_SIG               ((NBYTE_IN - NBYTE_OUT) * 8 - (int)__log2f(CUFFT_NX1)) // Not exact
#define TSAMP                 (NCHAN_KEEP_CHAN/(double)CUFFT_NX2)
#define NBIT                  8

#define SCL_INT8              127.0f          // int8_t
#define OFFS_INT8             0.0f
#define SCL_NSIG              6.0f            // 4 sigma, 99.993666%

namespace psrdada_cpp {
namespace effelsberg {
namespace paf {
namespace kernels {

__device__ __forceinline__ uint64_t swap64(uint64_t x)
{
    uint64_t result;
    uint2 t;
    asm("mov.b64 {%0,%1},%2; \n\t"
        : "=r"(t.x), "=r"(t.y) : "l"(x));
    t.x = __byte_perm(t.x, 0, 0x0123);
    t.y = __byte_perm(t.y, 0, 0x0123);
    asm("mov.b64 %0,{%1,%2}; \n\t"
        : "=l"(result) : "r"(t.y), "r"(t.x));
    return result;
}

__global__
void unpack_paf_to_float32(int64_t const* __restrict__ dbuf_in, 
    float2* __restrict__ dbuf_rt1, 
    uint64_t offset_rt1)
{
    uint64_t loc_in, loc_rt1;
    int64_t tmp;
    /* 
       Loc for the input array, it is in continuous order, it is in (STREAM_BUF_NDFSTP)T(NCHK_NIC)F(NSAMP_DF)T(NCHAN_CHK)F(NPOL_SAMP)P order
       This is for entire setting, since gridDim.z =1 and blockDim.z = 1, we can simply it to the latter format;
       Becareful here, if these number are not 1, we need to use a different format;
     */
    loc_in = blockIdx.x * gridDim.y * blockDim.x * blockDim.y +
        blockIdx.y * blockDim.x * blockDim.y +
        threadIdx.x * blockDim.y +
        threadIdx.y;
    tmp = BSWAP_64(dbuf_in[loc_in]);
    
    // Put the data into PFT order  
    loc_rt1 = blockIdx.y * gridDim.x * blockDim.x * blockDim.y +
        threadIdx.y * gridDim.x * blockDim.x +
        blockIdx.x * blockDim.x +
        threadIdx.x;
    
    dbuf_rt1[loc_rt1].x = (int16_t)((tmp & 0x000000000000ffffULL));  
    dbuf_rt1[loc_rt1].y = (int16_t)((tmp & 0x00000000ffff0000ULL) >> 16);
    
    loc_rt1 = loc_rt1 + offset_rt1;
    dbuf_rt1[loc_rt1].x = (int16_t)((tmp & 0x0000ffff00000000ULL) >> 32);
    dbuf_rt1[loc_rt1].y = (int16_t)((tmp & 0xffff000000000000ULL) >> 48);
}

} //namespace kernels


Unpacker::Unpacker(cudaStream_t stream)
    : _stream(stream)
{

}

Unpacker::~Unpacker()
{

}

void Unpacker::unpack(InputType const& input, OutputType& output)
{
    BOOST_LOG_TRIVIAL(debug) << "Unpacking PAF data";
    std::size_t output_size = input.size() * 2;
    BOOST_LOG_TRIVIAL(debug) << "Resizing output buffer to " << output_size << " elements";
    output.resize(output_size);

    std::size_t stream_ndf_chk = input.size() / NCHK_BEAM / NSAMP_DF / NCHAN_CHK;
    std::size_t offset_rt1 = stream_ndf_chk / 2;
    dim3 grid(stream_ndf_chk, NCHK_BEAM);
    dim3 block(NSAMP_DF, NCHAN_CHK);
    InputType::value_type const* input_ptr = thrust::raw_pointer_cast(input.data());
    OutputType::value_type* output_ptr = thrust::raw_pointer_cast(output.data());
    kernels::unpack_paf_to_float32<<< grid, block, 0, _stream>>>(
            input_ptr, output_ptr, offset_rt1);
    CUDA_ERROR_CHECK(cudaStreamSynchronize(_stream));
}


} //namespace paf
} //namespace effelsberg
} //namespace psrdada_cpp
