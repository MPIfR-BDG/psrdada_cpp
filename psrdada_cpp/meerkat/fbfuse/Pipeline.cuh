#ifndef PSRDADA_CPP_MEERKAT_FBFUSE_PIPELINE_CUH
#define PSRDADA_CPP_MEERKAT_FBFUSE_PIPELINE_CUH

#include "psrdada_cpp/meerkat/fbfuse/PipelineConfig.hpp"
#include "psrdada_cpp/meerkat/fbfuse/DelayManager.cuh"
#include "psrdada_cpp/meerkat/fbfuse/WeightsManager.cuh"
#include "psrdada_cpp/meerkat/fbfuse/SplitTranspose.cuh"
#include "psrdada_cpp/meerkat/fbfuse/CoherentBeamformer.cuh"
#include "psrdada_cpp/meerkat/fbfuse/IncoherentBeamformer.cuh"
#include "psrdada_cpp/dada_write_client.hpp"
#include "psrdada_cpp/double_device_buffer.cuh"
#include "psrdada_cpp/raw_bytes.hpp"
#include "cuda.h"
#include <memory>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {

class Pipeline
{
public:
    typedef thrust::device_vector<char2> VoltageVectorType;
    typedef thrust::device_vector<int8_t> PowerVectorType;
    typedef long double TimeType;

public:
    Pipeline(PipelineConfig const& config,
        DadaWriteClient& cb_writer,
        DadaWriteClient& ib_writer,
        std::size_t input_data_buffer_size);
    ~Pipeline();
    Pipeline(Pipeline const&) = delete;

    void init(RawBytes& header);
    bool operator()(RawBytes& data);

private:
    void process(VoltageVectorType const&, PowerVectorType&, PowerVectorType&);
    void set_header(RawBytes& header);

private:
    PipelineConfig const& _config;

    std::size_t _sample_clock_start;
    long double _sample_clock;
    long double _sync_time;
    long double _unix_timestamp;
    std::size_t _sample_clock_tick_per_block;
    std::size_t _call_count;

    DoubleDeviceBuffer<char2> _taftp_db; // Input from F-engine
    DoubleDeviceBuffer<int8_t> _tbtf_db; // Output of coherent beamformer
    DoubleDeviceBuffer<int8_t> _tf_db; // Output of incoherent beamformer

    DadaWriteClient& _cb_writer;
    DadaWriteClient::HeaderStream& _cb_header_stream;
    DadaWriteClient::DataStream& _cb_data_stream;
    DadaWriteClient& _ib_writer;
    DadaWriteClient::HeaderStream& _ib_header_stream;
    DadaWriteClient::DataStream& _ib_data_stream;

    cudaStream_t _h2d_copy_stream;
    cudaStream_t _processing_stream;
    cudaStream_t _d2h_copy_stream;

    std::size_t _nheap_groups_per_block;
    std::size_t _nsamples_per_dada_block;

    std::unique_ptr<DelayManager> _delay_manager;
    std::unique_ptr<WeightsManager> _weights_manager;
    std::unique_ptr<SplitTranspose> _split_transpose;
    std::unique_ptr<CoherentBeamformer> _coherent_beamformer;
    std::unique_ptr<IncoherentBeamformer> _incoherent_beamformer;

    VoltageVectorType _split_transpose_output;


};

} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

#endif //PSRDADA_CPP_MEERKAT_FBFUSE_PIPELINE_CUH
