#ifndef PSRDADA_CPP_EFFELSBERG_EDD_VLBI_CUH
#define PSRDADA_CPP_EFFELSBERG_EDD_VLBI_CUH

#include "psrdada_cpp/effelsberg/edd/Unpacker.cuh"
#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/cuda_utils.hpp"
#include "psrdada_cpp/double_device_buffer.cuh"
#include "psrdada_cpp/double_host_buffer.cuh"


#include <thrust/device_vector.h>


namespace psrdada_cpp {
namespace effelsberg {
namespace edd {

const size_t vlbiHeaderSize = 8 * 32 / 8; // bytes [8 words a 32 bit]


/// class VDIFHeaderView provides interprets a data block as VDIF compliant header 
/// See https://vlbi.org/vlbi-standards/vdif/specification 1.1.1 from June 2014 for details.
class VDIFHeaderView
{
  private:
    const uint32_t *data;
  public:
    VDIFHeaderView(const uint32_t* data);
    void setDataLocation(const uint32_t* _data);
    const uint32_t* getDataLocation() const;
    uint32_t getVersionNumber() const;
		bool isValid() const;
		uint32_t getSecondsFromReferenceEpoch() const;
		uint32_t getReferenceEpoch() const;
    size_t getTimestamp() const;
		uint32_t getDataFrameNumber() const;
		uint32_t getDataFrameLength() const;
		uint32_t getNumberOfChannels() const;
    bool isRealDataType() const;
		bool isComplexDataType() const;
		uint32_t getBitsPerSample() const;
		uint32_t getThreadId() const;
		uint32_t getStationId() const;

};


/// class VDIFHeader stores a VDIF compliant header block with conveniant
/// setters and getters. See https://vlbi.org/vlbi-standards/vdif/
/// specification 1.1.1 from June 2014 for details.
class VDIFHeader : public VDIFHeaderView
{
	private:
	  uint32_t data[8];

	public:
		VDIFHeader();
		VDIFHeader(const VDIFHeader &v);
		VDIFHeader& operator=(const VDIFHeader& other);

    // return pointer to the data block for low level manipulation
		uint32_t* getData();
		void setInvalid();
		void setValid();
		void setSecondsFromReferenceEpoch(uint32_t value);
		void setReferenceEpoch(uint32_t value);

    /// set reference epoch and seconds from reference epoch from POSIX time
    /// stamp
    void setTimeReferencesFromTimestamp(size_t);
    /// converts time reference data to POSIX time
		void setDataFrameNumber(uint32_t value);
		void setDataFrameLength(uint32_t value);
		void setNumberOfChannels(uint32_t value);
		void setComplexDataType();
		void setRealDataType();
	  void setBitsPerSample(uint32_t value);
	  void setThreadId(uint32_t value);
	  void setStationId(uint32_t value);
};





/**
 @class VLBI
 @brief Convert data to 2bit data in VDIF format.
 */
template <class HandlerType> class VLBI{
public:
  typedef uint64_t RawVoltageType;

public:
  /**
   * @brief      Constructor
   *
   * @param      buffer_bytes A RawBytes object wrapping a DADA header buffer
   * @param      input_bitDepth Bit depth of the sampled signal.
   * @param      speadHeapSize Size of the spead heap block.
   * @param      digitizer_threshold Threshold for the 2 bit digitization in units of the RMS of the signal.
   * @param      VDIFHeader Header of the VDIF output to be sed. Must contain size of the output VDIF payload in bytes.
   * @param      handler Output handler
   *
   */
  VLBI(std::size_t buffer_bytes, std::size_t input_bitDepth,
                    std::size_t speadHeapSize,
                    double sampleRate,
                    double digitizer_threshold,
                    const VDIFHeader &vdifHeader,
                    HandlerType &handler);
  ~VLBI();

  /**
   * @brief      A callback to be called on connection
   *             to a ring buffer.
   *
   * @detail     The first available header block in the
   *             in the ring buffer is provided as an argument.
   *             It is here that header parameters could be read
   *             if desired.
   *
   * @param      block  A RawBytes object wrapping a DADA header buffer
   */
  void init(RawBytes &block);

  /**
   * @brief      A callback to be called on acqusition of a new
   *             data block.
   *
   * @param      block  A RawBytes object wrapping a DADA data buffer output
   *             are the integrated specttra with/without bit set.
   */
  bool operator()(RawBytes &block);

private:
  std::size_t _buffer_bytes;
  std::size_t _input_bitDepth;
  std::size_t _output_bitDepth;
  std::size_t _speadHeapSize;
  std::size_t _outputBlockSize;
  double _sampleRate, _digitizer_threshold;
  VDIFHeader _vdifHeader;

  HandlerType &_handler;
  int _call_count;
  std::unique_ptr<Unpacker> _unpacker;

  // Input data
  DoubleDeviceBuffer<RawVoltageType> _raw_voltage_db;

  // Tmp data for processing
  thrust::device_vector<float> _unpacked_voltage;
  thrust::device_vector<float> _baseLineN;
  thrust::device_vector<float> _stdDevN;

  // Output data
  DoubleDeviceBuffer<uint8_t> _packed_voltage;
  DoublePinnedHostBuffer<uint8_t> _outputBuffer;

  // spill over between two dada blocks as vdif block not necessarily aligned
  thrust::host_vector<uint8_t, thrust::system::cuda::experimental::pinned_allocator<uint8_t>> _spillOver;

  cudaStream_t _h2d_stream;
  cudaStream_t _proc_stream;
  cudaStream_t _d2h_stream;
};


// pack float to 2 bit integers with VLBI non linear scaling with levels
// -n * sigma, -1 signa, sigma, n * sigma
// For performance/technical reasons it is
// sigma = \sqrt(sigma2)) and meanN = N * mean
__global__ void pack2bit_nonLinear(const float *__restrict__ input,
                         uint32_t *__restrict__ output, size_t inputSize,
                         float n, float *sigma2, float *meanN);

} //namespace edd
} //namespace effelsberg
} //namespace psrdada_cpp
#include "psrdada_cpp/effelsberg/edd/detail/VLBI.cu"

#endif // PSRDADA_CPP_EFFELSBERG_EDD_VLBI_CUH



