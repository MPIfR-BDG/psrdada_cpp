#include "psrdada_cpp/meerkat/tools/feng_to_dada.cuh"
#include "psrdada_cpp/cuda_utils.hpp"

namespace psrdada_cpp {
namespace meerkat {
namespace tools {

    template <class HandlerType>
    FengToDada<HandlerType>::FengToDada(std::size_t nchans, HandlerType& handler)
    : _nchans(nchans)
    , _handler(handler)
    {
    }

    template <class HandlerType>
    FengToDada<HandlerType>::~FengToDada()
    {
    }

    template <class HandlerType>
    void FengToDada<HandlerType>::init(RawBytes& block)
    {
        _handler.init(block);
    }

    template <class HandlerType>
    bool FengToDada<HandlerType>::operator()(RawBytes& block)
    {
        std::size_t used = block.used_bytes();
        std::size_t nbytes_per_timestamp =
            _nchans * MEERKAT_FENG_NSAMPS_PER_HEAP
            * MEERKAT_FENG_NPOL_PER_HEAP * sizeof(char2);
        if (used%nbytes_per_timestamp != 0)
        {
            throw std::runtime_error("Number of bytes in buffer is not an integer "
                "muliple of the number of bytes per timestamp");
        }
        std::size_t size = used/sizeof(int);
        int ntimestamps = used/nbytes_per_timestamp;
        BOOST_LOG_TRIVIAL(debug) << "Number of time heaps: " << ntimestamps;
        _input.resize(size);
        _output.resize(size);
        int* d_input_ptr = thrust::raw_pointer_cast(_input.data());
        int* d_output_ptr = thrust::raw_pointer_cast(_output.data());
        CUDA_ERROR_CHECK(cudaMemcpy(d_input_ptr, block.ptr(), used, cudaMemcpyHostToDevice));
        kernels::feng_heaps_to_dada<<<ntimestamps, MEERKAT_FENG_NSAMPS_PER_HEAP>>>
            (d_input_ptr, d_output_ptr, _nchans);
        CUDA_ERROR_CHECK(cudaDeviceSynchronize());
        thrust::copy(_output.begin(), _output.end(), (int*) block.ptr());
        _handler(block);
        return false;
    }

} //tools
} //meerkat
} //psrdada_cpp