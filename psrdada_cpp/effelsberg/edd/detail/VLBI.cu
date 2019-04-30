#include "psrdada_cpp/effelsberg/edd/VLBI.cuh"

//#include "psrdada_cpp/effelsberg/edd/GatedSpectrometer.cuh"
#include "psrdada_cpp/cuda_utils.hpp"

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <thrust/system/cuda/execution_policy.h>

#include <iostream>
#include <cstring>
#include <sstream>

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {


template <class HandlerType>
VLBI<HandlerType>::VLBI(
    std::size_t buffer_bytes,
    std::size_t input_bitDepth,
    std::size_t speadHeapSize,
    std::size_t outputBlockSize,
    HandlerType &handler)
    : _buffer_bytes(buffer_bytes),
      _input_bitDepth(input_bitDepth),
      _outputBlockSize(outputBlockSize),
      _output_bitDepth(2),
      _speadHeapSize(speadHeapSize),
      _handler(handler), 
      _call_count(0) {

  // Sanity checks
  // check for any device errors
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());

  BOOST_LOG_TRIVIAL(info) << "Creating new VLBI instance";
  BOOST_LOG_TRIVIAL(info) << " Output data in VDIF format with " << vlbiHeaderSize << "bytes header info and " << outputBlockSize << " bytes payload";
  BOOST_LOG_TRIVIAL(debug) << " Expecting speadheaps of size " << speadHeapSize << " byte";

  std::size_t n64bit_words = _buffer_bytes / sizeof(uint64_t);
  BOOST_LOG_TRIVIAL(debug) << "Allocating memory";
  _raw_voltage_db.resize(n64bit_words);
  BOOST_LOG_TRIVIAL(debug) << "  Input voltages size (in 64-bit words): "
                           << _raw_voltage_db.size();

  _packed_voltage.resize(n64bit_words * 64 / input_bitDepth / 4);

  _spillOver.reserve(5000);
  BOOST_LOG_TRIVIAL(debug) << "  Output voltages size: " << _packed_voltage.size() << " byte";

  CUDA_ERROR_CHECK(cudaStreamCreate(&_h2d_stream));
  CUDA_ERROR_CHECK(cudaStreamCreate(&_proc_stream));
  CUDA_ERROR_CHECK(cudaStreamCreate(&_d2h_stream));

  _unpacker.reset(new Unpacker(_proc_stream));

  _vdifHeader.setBitsPerSample(2);
  _vdifHeader.setNumberOfChannels(1);
  _vdifHeader.setRealDataType();

  //_vdifHeader.setThreadId(threadId);
  //_vdifHeader.setStationId(stationId);
  _vdifHeader.setDataFrameLength(outputBlockSize);
  //_vdifHeader.setReferenceEpoch(referenceEpoch);
  //_vdifHeader.setSecondsFromReferenceEpoch(secondsFromReferenceEpoch_sync);


} // constructor


template <class HandlerType>
VLBI<HandlerType>::~VLBI() {
  BOOST_LOG_TRIVIAL(debug) << "Destroying VLBI";
  cudaStreamDestroy(_h2d_stream);
  cudaStreamDestroy(_proc_stream);
  cudaStreamDestroy(_d2h_stream);
}


template <class HandlerType>
void VLBI<HandlerType>::init(RawBytes &block) {
  BOOST_LOG_TRIVIAL(debug) << "VLBI init called";
  std::stringstream headerInfo;
  headerInfo << "\n" << "# VLBI parameters: \n";

  size_t bEnd = std::strlen(block.ptr());
  if (bEnd + headerInfo.str().size() < block.total_bytes())
  {
    std::strcpy(block.ptr() + bEnd, headerInfo.str().c_str());
  }
  else
  {
    BOOST_LOG_TRIVIAL(warning) << "Header of size " << block.total_bytes()
      << " bytes already contains " << bEnd
      << "bytes. Cannot add gated spectrometer info of size "
      << headerInfo.str().size() << " bytes.";
  }

  _handler.init(block);
}


template <class HandlerType>
bool VLBI<HandlerType>::operator()(RawBytes &block) {
  ++_call_count;
  BOOST_LOG_TRIVIAL(debug) << "VLBI operator() called (count = "
                           << _call_count << ")";
  if (block.used_bytes() != _buffer_bytes) { /* Unexpected buffer size */
    BOOST_LOG_TRIVIAL(error) << "Unexpected Buffer Size - Got "
                             << block.used_bytes() << " byte, expected "
                             << _buffer_bytes << " byte)";
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    cudaProfilerStop();
    return true;
  }
  ////////////////////////////////////////////////////////////////////////
  // Copy data to device
  CUDA_ERROR_CHECK(cudaStreamSynchronize(_h2d_stream));
  _raw_voltage_db.swap();

  BOOST_LOG_TRIVIAL(debug) << "   block.used_bytes() = " << block.used_bytes()
                           << ", dataBlockBytes = " << _buffer_bytes<< "\n";

  CUDA_ERROR_CHECK(cudaMemcpyAsync(static_cast<void *>(_raw_voltage_db.a_ptr()),
                                   static_cast<void *>(block.ptr()),
                                   _buffer_bytes, cudaMemcpyHostToDevice,
                                   _h2d_stream));
  if (_call_count == 1) {
    return false;
  }
  ////////////////////////////////////////////////////////////////////////
  // Process data
  _packed_voltage.swap();

  BOOST_LOG_TRIVIAL(debug) << "Unpacking raw voltages";
  switch (_input_bitDepth) {
  case 8:
    _unpacker->unpack<8>(_raw_voltage_db.b(), _unpacked_voltage);
    break;
  case 12:
    _unpacker->unpack<12>(_raw_voltage_db.b(), _unpacked_voltage);
    break;
  default:
    throw std::runtime_error("Unsupported number of bits");
  }

  // ToDo: Eventually calulate minV, maxV from mean and std.

  float minV = -2;
  float maxV = 2;
  pack_2bit(_unpacked_voltage, _packed_voltage.b(), minV, maxV, _proc_stream);


  CUDA_ERROR_CHECK(cudaStreamSynchronize(_proc_stream));

  if ((_call_count == 2)) {
    return false;
  }
  _outputBuffer.swap();

  ////////////////////////////////////////////////////////////////////////
  BOOST_LOG_TRIVIAL(debug) << "Copy Data back to host";
  CUDA_ERROR_CHECK(cudaStreamSynchronize(_d2h_stream));

  size_t remainingBytes = _outputBlockSize - _spillOver.size();
  size_t numberOfBlocksInOutput = (_packed_voltage.size() - remainingBytes) / _outputBlockSize;
  BOOST_LOG_TRIVIAL(debug) << " Number of blocks in output" << numberOfBlocksInOutput;

  _outputBuffer.a().resize((1+numberOfBlocksInOutput) * (_outputBlockSize + vlbiHeaderSize));

  BOOST_LOG_TRIVIAL(debug) << " Copying " << _spillOver.size() << " bytes spill over";
  // leave room for header and fill first block of output with spill over
  std::copy(_spillOver.begin(), _spillOver.end(), _outputBuffer.a().begin() + vlbiHeaderSize);

  BOOST_LOG_TRIVIAL(debug) << " Copying remaining " << remainingBytes << " bytes for first block";
  // cuda memcopy remainder of first block
  CUDA_ERROR_CHECK(cudaMemcpyAsync(static_cast<void *>(_packed_voltage.a_ptr()),
                        static_cast<void *>(_outputBuffer.a_ptr()),
                        remainingBytes,
                        cudaMemcpyDeviceToHost, _d2h_stream));

  const size_t dpitch = _outputBlockSize + vlbiHeaderSize;
  const size_t spitch = _outputBlockSize;
  const size_t width = _outputBlockSize;
  size_t height = numberOfBlocksInOutput;

  BOOST_LOG_TRIVIAL(debug) << " Copying " << numberOfBlocksInOutput << " blocks a " << _outputBlockSize << " bytes";
  // we now have a full first block, pitch copy rest leaving room for the header
  CUDA_ERROR_CHECK(cudaMemcpy2DAsync((void*) (_outputBuffer.a_ptr()+ _outputBlockSize + 2 * vlbiHeaderSize) , dpitch, (void*) thrust::raw_pointer_cast(_packed_voltage.a_ptr() + remainingBytes), spitch, width, height, cudaMemcpyDeviceToHost, _d2h_stream));


  // new spill over
  _spillOver.resize(_packed_voltage.size()  - remainingBytes - numberOfBlocksInOutput * _outputBlockSize);

  size_t offset = numberOfBlocksInOutput * _outputBlockSize + remainingBytes;
  BOOST_LOG_TRIVIAL(debug) << " New spill over size " <<  _spillOver.size() << " bytes with offset " << offset;

  CUDA_ERROR_CHECK(cudaMemcpyAsync(static_cast<void *>(_packed_voltage.a_ptr() + offset),
                        static_cast<void *>( thrust::raw_pointer_cast(_spillOver.data()) ),
                        _spillOver.size(),
                        cudaMemcpyDeviceToHost, _d2h_stream));

  // fill in header data
  for (size_t i = 0; i < numberOfBlocksInOutput + 1; i ++)
  {
    _vdifHeader.setDataFrameNumber(i); // ToDo: Use correct number
    _vdifHeader.setSecondsFromReferenceEpoch(_call_count); // ToDo use correct number

    std::copy(reinterpret_cast<uint8_t* >(_vdifHeader.getData()),
        reinterpret_cast<uint8_t* >(_vdifHeader.getData()) + vlbiHeaderSize,
        _outputBuffer.a().begin() + i * (_outputBlockSize + vlbiHeaderSize));
  }

  if (_call_count == 3) {
    return false;
  }


  // Wrap in a RawBytes object here;
  RawBytes bytes(reinterpret_cast<char *>(_outputBuffer.b_ptr()),
                 _outputBuffer.b().size(),
                 _outputBuffer.b().size());
  BOOST_LOG_TRIVIAL(debug) << "Calling handler, processing " << _outputBuffer.b().size() << " bytes";
  // The handler can't do anything asynchronously without a copy here
  // as it would be unsafe (given that it does not own the memory it
  // is being passed).

  _handler(bytes);
  return false; //
} // operator ()

} // edd
} // effelsberg
} // psrdada_cpp

