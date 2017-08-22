#include "psrdada_cpp/meerkat/tools/feng_to_bandpass.cuh"
#include "psrdada_cpp/meerkat/constants.hpp"
#include "psrdada_cpp/cuda_utils.hpp"
#include "psrdada_cpp/common.hpp"
#include "cuComplex.h"
#include <fstream>
#include <iomanip>

namespace psrdada_cpp {
namespace meerkat {
namespace tools {

    template <class HandlerType>
    FengToBandpass<HandlerType>::FengToBandpass(std::size_t nchans, std::size_t nants, HandlerType& handler)
    : _nchans(nchans)
    , _natnennas(nants)
    , _handler(handler)
    {
        //Will output data as an array of bandpasses for each
        //polarisation and antenna
        _output.resize(_nchans * _natnennas * MEERKAT_FENG_NPOL_PER_HEAP);
    }

    template <class HandlerType>
    FengToBandpass<HandlerType>::~FengToBandpass()
    {
    }

    template <class HandlerType>
    void FengToBandpass<HandlerType>::init(RawBytes& block)
    {
        _handler.init(block);
    }

    template <class HandlerType>
    bool FengToBandpass<HandlerType>::operator()(RawBytes& block)
    {
        std::size_t used = block.used_bytes();
        std::size_t nbytes_per_timestamp =
            _natnennas * _nchans * MEERKAT_FENG_NSAMPS_PER_HEAP
            * MEERKAT_FENG_NPOL_PER_HEAP * sizeof(char2);
        if (used%nbytes_per_timestamp != 0)
        {
            throw std::runtime_error("Number of bytes in buffer is not an integer "
                "muliple of the number of bytes per timestamp");
        }
        std::size_t size = used/sizeof(char2);
        int ntimestamps = used/nbytes_per_timestamp;
        _input.resize(size);
        char2* d_input_ptr = thrust::raw_pointer_cast(_input.data());
        float* d_output_ptr = thrust::raw_pointer_cast(_output.data());
        CUDA_ERROR_CHECK(cudaMemcpy(d_input_ptr, block.ptr(), used, cudaMemcpyHostToDevice));
        dim3 grid(_natnennas,_nchans,MEERKAT_FENG_NPOL_PER_HEAP);
        kernels::feng_heaps_to_bandpass<<<grid,MEERKAT_FENG_NSAMPS_PER_HEAP>>>
            (d_input_ptr, d_output_ptr, _nchans, _natnennas, ntimestamps);
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());
        thrust::copy(_output.begin(),_output.end(),(float*) block.ptr());
        block.used_bytes(_output.size() * sizeof(float));
        _handler(block);
        return false;
    }
}
}
}