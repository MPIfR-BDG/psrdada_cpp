#include "psrdada_cpp/effelsberg/edd/VLBI.cuh"
#include "psrdada_cpp/effelsberg/edd/Packer.cuh"
#include "psrdada_cpp/effelsberg/edd/Tools.cuh"
#include "psrdada_cpp/cuda_utils.hpp"

#include "ascii_header.h" // dada

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/extrema.h>

#include <cstring>
#include <iostream>
#include <sstream>

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {


template <class HandlerType>
VLBI<HandlerType>::VLBI(std::size_t buffer_bytes, std::size_t input_bitDepth,
                        std::size_t speadHeapSize, double sampleRate,
                        double digitizer_threshold,
                        const VDIFHeader &vdifHeader, HandlerType &handler)
    : _buffer_bytes(buffer_bytes), _input_bitDepth(input_bitDepth),
      _sampleRate(sampleRate), _digitizer_threshold(digitizer_threshold),
      _vdifHeader(vdifHeader), _output_bitDepth(2),
      _speadHeapSize(speadHeapSize), _handler(handler), _call_count(0) {

  // Sanity checks
  // check for any device errors
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());

  BOOST_LOG_TRIVIAL(info) << "Creating new VLBI instance";
  BOOST_LOG_TRIVIAL(info) << "   Output data in VDIF format with "
                          << vlbiHeaderSize << "bytes header info and "
                          << _vdifHeader.getDataFrameLength() * 8
                          << " bytes data frame length";
  BOOST_LOG_TRIVIAL(debug) << "   Expecting speadheaps of size "
                           << speadHeapSize << "   byte";

  BOOST_LOG_TRIVIAL(debug) << "   Sample rate " << _sampleRate << " Hz";

  std::size_t n64bit_words = _buffer_bytes / sizeof(uint64_t);
  BOOST_LOG_TRIVIAL(debug) << "Allocating memory";
  _raw_voltage_db.resize(n64bit_words);
  BOOST_LOG_TRIVIAL(debug) << "   Input voltages size : "
                           << _raw_voltage_db.size() << " 64-bit words,"
                           << _raw_voltage_db.size() * 64 / 8 << " bytes";
  _unpacked_voltage.resize(n64bit_words * 64 / input_bitDepth );
  _packed_voltage.resize(n64bit_words * 64 / input_bitDepth * _output_bitDepth /
                         8);
  BOOST_LOG_TRIVIAL(debug) << "   Output voltages size: "
                           << _packed_voltage.size() << " byte";
  _spillOver.reserve(vdifHeader.getDataFrameLength() * 8 - vlbiHeaderSize);

  // number of vlbi frames per input block
  size_t nSamplesPerInputBlock = _packed_voltage.size() * 8 / _output_bitDepth;
  size_t frames_per_block = _packed_voltage.size() / (vdifHeader.getDataFrameLength() * 8 - vlbiHeaderSize);
  BOOST_LOG_TRIVIAL(debug) << "   this correspoonds to " << frames_per_block << " - " << frames_per_block + 1 << " frames";

  _outputBuffer.resize((frames_per_block+1) * vdifHeader.getDataFrameLength() * 8 );
  // potetnitally invalidating the last frame
  BOOST_LOG_TRIVIAL(info) << "   Output data in VDIF format with " << _outputBuffer.size() << " bytes per buffer";



  CUDA_ERROR_CHECK(cudaStreamCreate(&_h2d_stream));
  CUDA_ERROR_CHECK(cudaStreamCreate(&_proc_stream));
  CUDA_ERROR_CHECK(cudaStreamCreate(&_d2h_stream));

  _unpacker.reset(new Unpacker(_proc_stream));

  _vdifHeader.setBitsPerSample(_output_bitDepth - 1); // bits per sample - 1
  _vdifHeader.setRealDataType();
} // constructor


template <class HandlerType> VLBI<HandlerType>::~VLBI() {
  BOOST_LOG_TRIVIAL(debug) << "Destroying VLBI";
  cudaStreamDestroy(_h2d_stream);
  cudaStreamDestroy(_proc_stream);
  cudaStreamDestroy(_d2h_stream);
}


template <class HandlerType> void VLBI<HandlerType>::init(RawBytes &block) {
  BOOST_LOG_TRIVIAL(debug) << "VLBI init called";

  size_t sync_time = 0;
  if (ascii_header_get(block.ptr(), "SYNC_TIME", "%ld", &sync_time) != 1)
  {
    BOOST_LOG_TRIVIAL(warning) << "No or multiple SYNC_TIME parameters in header stream! Not setting reference";
    return;
  }
  size_t sample_clock_start = 0;
  if (ascii_header_get(block.ptr(), "SAMPLE_CLOCK_START", "%ld", &sample_clock_start) != 1)
  {
    BOOST_LOG_TRIVIAL(warning) << "No or multiple SAMPLE_CLOCK_START in header stream! Not setting reference time";
    return;
  }

  size_t timestamp = sync_time + sample_clock_start / _sampleRate;
  BOOST_LOG_TRIVIAL(info) << "POSIX timestamp  captured from header: " << timestamp << " = " << sync_time << " + " << sample_clock_start << " / " << _sampleRate << " = SYNC_TIME + SAMPLE_CLOCK_START/SAMPLERATE" ;
  _vdifHeader.setTimeReferencesFromTimestamp(timestamp);

  std::stringstream headerInfo;
  headerInfo << "\n"
             << "# VLBI parameters: \n";

  size_t bEnd = std::strlen(block.ptr());
  if (bEnd + headerInfo.str().size() < block.total_bytes()) {
    std::strcpy(block.ptr() + bEnd, headerInfo.str().c_str());
  } else {
    BOOST_LOG_TRIVIAL(warning) << "Header of size " << block.total_bytes()
                               << " bytes already contains " << bEnd
                               << "bytes. Cannot add VLBI info of size "
                               << headerInfo.str().size() << " bytes.";
  }

  _baseLineN.resize(array_sum_Nthreads);
  _stdDevN.resize(array_sum_Nthreads);

  _handler.init(block);
}


template <class HandlerType>
bool VLBI<HandlerType>::operator()(RawBytes &block) {
  ++_call_count;
  BOOST_LOG_TRIVIAL(debug) << "VLBI operator() called (count = " << _call_count
                           << ")";
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
                           << ", dataBlockBytes = " << _buffer_bytes << "\n";

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


  BOOST_LOG_TRIVIAL(debug) << "Calculate baseline";
  psrdada_cpp::effelsberg::edd::
      array_sum<<<64, array_sum_Nthreads, 0, _proc_stream>>>(
          thrust::raw_pointer_cast(_unpacked_voltage.data()),
          _unpacked_voltage.size(),
          thrust::raw_pointer_cast(_baseLineN.data()));
  psrdada_cpp::effelsberg::edd::
      array_sum<<<1, array_sum_Nthreads, 0, _proc_stream>>>(
          thrust::raw_pointer_cast(_baseLineN.data()), _baseLineN.size(),
          thrust::raw_pointer_cast(_baseLineN.data()));

  BOOST_LOG_TRIVIAL(debug) << "Calculate std-dev";
  psrdada_cpp::effelsberg::edd::
      scaled_square_residual_sum<<<64, array_sum_Nthreads, 0, _proc_stream>>>(
          thrust::raw_pointer_cast(_unpacked_voltage.data()),
          _unpacked_voltage.size(), thrust::raw_pointer_cast(_baseLineN.data()),
          thrust::raw_pointer_cast(_stdDevN.data()));
  psrdada_cpp::effelsberg::edd::
      array_sum<<<1, array_sum_Nthreads, 0, _proc_stream>>>(
          thrust::raw_pointer_cast(_stdDevN.data()), _stdDevN.size(),
          thrust::raw_pointer_cast(_stdDevN.data()));


  // non linear packing
  BOOST_LOG_TRIVIAL(debug) << "Packing data with non linear 2-bit packaging "
                              "using levels -v*sigma, 0, v*sigma with v = "
                           << _digitizer_threshold;
  _packed_voltage.b().resize(_unpacked_voltage.size() * 2 / 8);
  BOOST_LOG_TRIVIAL(debug) << "Input size: " << _unpacked_voltage.size()
                           << " elements";
  BOOST_LOG_TRIVIAL(debug) << "Resizing output buffer to "
                           << _packed_voltage.b().size() << " byte";

  pack2bit_nonLinear<<<128, 1024, 0, _proc_stream>>>(
      thrust::raw_pointer_cast(_unpacked_voltage.data()),
      (uint32_t *)thrust::raw_pointer_cast(_packed_voltage.b().data()),
      _unpacked_voltage.size(), _digitizer_threshold,
      thrust::raw_pointer_cast(_stdDevN.data()),
      thrust::raw_pointer_cast(_baseLineN.data()));

  CUDA_ERROR_CHECK(cudaStreamSynchronize(_proc_stream));
  BOOST_LOG_TRIVIAL(trace) << " Standard Deviation squared: " << _stdDevN[0]
                           << " "
                           << "Mean Value: "
                           << _baseLineN[0] / _unpacked_voltage.size();

  if ((_call_count == 2)) {
    return false;
  }
  _outputBuffer.swap();

  ////////////////////////////////////////////////////////////////////////
  BOOST_LOG_TRIVIAL(debug) << "Copy Data back to host";
  CUDA_ERROR_CHECK(cudaStreamSynchronize(_d2h_stream));

  const size_t outputBlockSize = _vdifHeader.getDataFrameLength() * 8 - vlbiHeaderSize;

  const size_t totalSizeOfData = _packed_voltage.size() + _spillOver.size(); // current array + remaining of previous

  size_t numberOfBlocksInOutput = totalSizeOfData / outputBlockSize;

  size_t remainingBytes = outputBlockSize - _spillOver.size();
  BOOST_LOG_TRIVIAL(debug) << "   Number of blocks in output "
                           << numberOfBlocksInOutput;

  //_outputBuffer.a().resize(numberOfBlocksInOutput *
   //                        (outputBlockSize + vlbiHeaderSize));

  BOOST_LOG_TRIVIAL(debug) << "   Copying " << _spillOver.size()
                           << " bytes spill over";
  // leave room for header and fill first block of output with spill over
  std::copy(_spillOver.begin(), _spillOver.end(),
            _outputBuffer.a().begin() + vlbiHeaderSize);

  BOOST_LOG_TRIVIAL(debug) << "   Copying remaining " << remainingBytes
                           << " bytes for first block";
  // cuda memcopy remainder of first block
  CUDA_ERROR_CHECK(cudaMemcpyAsync(static_cast<void *>(_outputBuffer.a_ptr() + vlbiHeaderSize + _spillOver.size()),
                                   static_cast<void *>(_packed_voltage.a_ptr()),
                                   remainingBytes, cudaMemcpyDeviceToHost,
                                   _d2h_stream));

  const size_t dpitch = outputBlockSize + vlbiHeaderSize;
  const size_t spitch = outputBlockSize;
  const size_t width = outputBlockSize;
  size_t height = numberOfBlocksInOutput-1;

  BOOST_LOG_TRIVIAL(debug) << "   Copying " << height
                           << " blocks a " << outputBlockSize << " bytes";
  // we now have a full first block, pitch copy rest leaving room for the header
  CUDA_ERROR_CHECK(cudaMemcpy2DAsync(
      (void *)(_outputBuffer.a_ptr() + outputBlockSize + 2 * vlbiHeaderSize),
      dpitch, (void *)thrust::raw_pointer_cast(_packed_voltage.a_ptr() +
                                               remainingBytes),
      spitch, width, height, cudaMemcpyDeviceToHost, _d2h_stream));


  // new spill over
  _spillOver.resize(totalSizeOfData - numberOfBlocksInOutput * outputBlockSize);

  size_t offset = (numberOfBlocksInOutput-1) * outputBlockSize + remainingBytes;
  BOOST_LOG_TRIVIAL(debug) << " New spill over size " << _spillOver.size()
                           << " bytes with offset " << offset;

  CUDA_ERROR_CHECK(cudaMemcpyAsync(
      static_cast<void *>(thrust::raw_pointer_cast(_spillOver.data())),
      static_cast<void *>(_packed_voltage.a_ptr() + offset),
      _spillOver.size(), cudaMemcpyDeviceToHost, _d2h_stream));

  // fill in header data
  const uint32_t samplesPerDataFrame = outputBlockSize * 8 / _output_bitDepth;
  const uint32_t dataFramesPerSecond = _sampleRate / samplesPerDataFrame;

  BOOST_LOG_TRIVIAL(debug) << " Samples per data frame: " << samplesPerDataFrame;
  BOOST_LOG_TRIVIAL(debug) << " Dataframes per second: " << dataFramesPerSecond;

  for (uint32_t ib = 0; ib < _outputBuffer.a().size(); ib += _vdifHeader.getDataFrameLength() * 8)
  {
     // copy header to correct position
    std::copy(reinterpret_cast<uint8_t *>(_vdifHeader.getData()),
        reinterpret_cast<uint8_t *>(_vdifHeader.getData()) + vlbiHeaderSize,
        _outputBuffer.a().begin() + ib);
    size_t i = ib / _vdifHeader.getDataFrameLength() / 8;

    // invalidate rest of data so it can be dropped later.
    // Needed so that the outpuitbuffer can have always the same size
    if (i < numberOfBlocksInOutput)
    {
      _vdifHeader.setValid();
    }
    else
    {
      _vdifHeader.setInvalid();
      continue;
    }

    // update header
    uint32_t dataFrame = _vdifHeader.getDataFrameNumber();
    if (i < 5)
      BOOST_LOG_TRIVIAL(debug) << i << " Dataframe Number: " << dataFrame;
    if (dataFrame < dataFramesPerSecond)
    {
      _vdifHeader.setDataFrameNumber(dataFrame + 1);
    }
    else
    {
      _vdifHeader.setDataFrameNumber(0);
      _vdifHeader.setSecondsFromReferenceEpoch(_vdifHeader.getSecondsFromReferenceEpoch() + 1);
    }
  }

  if (_call_count == 3) {
    return false;
  }

  // Wrap in a RawBytes object here;
  RawBytes bytes(reinterpret_cast<char *>(_outputBuffer.b_ptr()),
                 _outputBuffer.b().size(), _outputBuffer.b().size());
  BOOST_LOG_TRIVIAL(debug) << "Calling handler, processing "
                           << _outputBuffer.b().size() << " bytes";
  // The handler can't do anything asynchronously without a copy here
  // as it would be unsafe (given that it does not own the memory it
  // is being passed).

  _handler(bytes);
  return false; //
} // operator ()

} // edd
} // effelsberg
} // psrdada_cpp

