#include "psrdada_cpp/effelsberg/edd/VLBI.cuh"
#include "psrdada_cpp/effelsberg/edd/Tools.cuh"

#include "psrdada_cpp/cuda_utils.hpp"
#include <cuda.h>
#include <cstdint>

#include "boost/date_time/posix_time/posix_time.hpp"
#include "boost/date_time/gregorian/gregorian.hpp"

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {



VDIFHeaderView::VDIFHeaderView(const uint32_t* data) : data(data) {};
void VDIFHeaderView::setDataLocation(const uint32_t* _data) {
  data = _data;
};

bool VDIFHeaderView::isValid() const {
  return (getBitsValue(data[0], 31, 31) == 0);
}

uint32_t VDIFHeaderView::getSecondsFromReferenceEpoch() const {
  return getBitsValue(data[0], 0, 29);
}

uint32_t VDIFHeaderView::getReferenceEpoch() const {
  return getBitsValue(data[1], 24, 29);
}

uint32_t VDIFHeaderView::getDataFrameNumber() const {
  return getBitsValue(data[1], 0, 23);
}
uint32_t VDIFHeaderView::getDataFrameLength() const {
  return getBitsValue(data[2], 0, 23);
}

uint32_t VDIFHeaderView::getVersionNumber() const {
  return getBitsValue(data[2], 29, 31);
}

uint32_t VDIFHeaderView::getNumberOfChannels() const {
  return getBitsValue(data[2], 24, 28);
}

bool VDIFHeaderView::isRealDataType() const {
  return (getBitsValue(data[3], 31, 31) == 0);
}

bool VDIFHeaderView::isComplexDataType() const {
  return (getBitsValue(data[3], 31, 31) == 1);
}

uint32_t VDIFHeaderView::getBitsPerSample() const {
  return getBitsValue(data[3], 26, 30);
}

uint32_t VDIFHeaderView::getThreadId() const {
  return getBitsValue(data[3], 16, 25);
}

uint32_t VDIFHeaderView::getStationId() const {
  return getBitsValue(data[3], 0, 15);
}

size_t VDIFHeaderView::getTimestamp() const {
  boost::gregorian::date vdifEpochBegin(getReferenceEpoch() / 2 + 2000,
                                        ((getReferenceEpoch() % 2) * 6) + 1, 1);
  boost::posix_time::ptime pt =  boost::posix_time::ptime(vdifEpochBegin) + boost::posix_time::seconds(getSecondsFromReferenceEpoch());
  boost::posix_time::ptime unixEpoch =
      boost::posix_time::time_from_string("1970-01-01 00:00:00.000");
  boost::posix_time::time_duration delta = pt - unixEpoch;
  return delta.total_seconds();
}


VDIFHeader::VDIFHeader() : VDIFHeaderView(data)
{
  for (int i = 0; i < 8; i++) {
    data[i] = 0U;
  }

  // set standard VDIF header
  setBitsWithValue(data[1], 30, 30, 0);
  setBitsWithValue(data[1], 30, 31, 0);

  // set Version Number to 1
  setBitsWithValue(data[2], 29, 31, 1);
}

uint32_t *VDIFHeader::getData() { return data; }

void VDIFHeader::setInvalid() { setBitsWithValue(data[0], 31, 31, 1); }

void VDIFHeader::setValid() { setBitsWithValue(data[0], 31, 31, 0); }

void VDIFHeader::setSecondsFromReferenceEpoch(uint32_t value) {
  setBitsWithValue(data[0], 0, 29, value);
}

void VDIFHeader::setReferenceEpoch(uint32_t value) {
  setBitsWithValue(data[1], 24, 29, value);
}

void VDIFHeader::setDataFrameNumber(uint32_t value) {
  setBitsWithValue(data[1], 0, 23, value);
}

void VDIFHeader::setDataFrameLength(uint32_t value) {
  setBitsWithValue(data[2], 0, 23, value);
}

void VDIFHeader::setNumberOfChannels(uint32_t value) {
  setBitsWithValue(data[2], 24, 28, value);
}

void VDIFHeader::setComplexDataType() { setBitsWithValue(data[3], 31, 31, 1); }

void VDIFHeader::setRealDataType() { setBitsWithValue(data[0], 31, 31, 0); }

void VDIFHeader::setBitsPerSample(uint32_t value) {
  setBitsWithValue(data[3], 26, 30, value);
}

void VDIFHeader::setThreadId(uint32_t value) {
  setBitsWithValue(data[3], 16, 25, value);
}

void VDIFHeader::setStationId(uint32_t value) {
  setBitsWithValue(data[3], 0, 15, value);
}

void VDIFHeader::setTimeReferencesFromTimestamp(size_t sync_time) {
  BOOST_LOG_TRIVIAL(debug) << "Setting time reference from timestamp: " << sync_time;
  boost::posix_time::ptime pt = boost::posix_time::from_time_t(sync_time);
  BOOST_LOG_TRIVIAL(debug) << "  - posix_time:  " << pt;

  boost::gregorian::date epochBegin(pt.date().year(),
                                    ((pt.date().month() <= 6) ? 1 : 7), 1);
  BOOST_LOG_TRIVIAL(debug) << "  - epochBegin: " << epochBegin;
  int refEpoch = (epochBegin.year() - 2000) * 2 + (epochBegin.month() >= 7);
  if (refEpoch < 0)
  {
    BOOST_LOG_TRIVIAL(error) << "Cannot encode time before 1 Jan 2000 - received " << pt;
  }
  BOOST_LOG_TRIVIAL(debug) << "  - reference epoch: " << refEpoch;
  setReferenceEpoch(refEpoch);

  boost::posix_time::time_duration delta =
      pt - boost::posix_time::ptime(epochBegin);
  BOOST_LOG_TRIVIAL(debug) << "  - time delta since epoch begin: " << delta << " = "  << delta.total_seconds();
  setSecondsFromReferenceEpoch(delta.total_seconds());

    BOOST_LOG_TRIVIAL(debug) << " Time stamp: " << sync_time
        << " => VDIF Reference epoch: " << getReferenceEpoch()
        << " at " << getSecondsFromReferenceEpoch() << " s";
}




__global__ void pack2bit_nonLinear(const float *__restrict__ input,
                                   uint32_t *__restrict__ output,
                                   size_t inputSize, float v, float *sigma2,
                                   float *meanN) {
  // number of values to pack into one output element, use 32 bit here to
  // maximize number of threads
  const uint8_t NPACK = 32 / 2;

  __shared__ uint32_t tmp[1024];
  const float vsigma = v * sqrt(*sigma2);

  for (uint32_t i = NPACK * blockIdx.x * blockDim.x + threadIdx.x;
       (i < inputSize); i += blockDim.x * gridDim.x * NPACK) {
    tmp[threadIdx.x] = 0;

    #pragma unroll
    for (uint8_t j = 0; j < NPACK; j++) {
      // Load new input value, clip and convert to Nbit integer
      const float inp = input[i + j * blockDim.x] - (*meanN) / inputSize;

      uint32_t p = 0;
      p += (inp >= (-1. * vsigma));
      p += (inp >= (0));
      p += (inp >= (1. * vsigma));
      // this is more efficient than fmin, fmax for clamp and cast.

      // store in shared memory with linear access
      tmp[threadIdx.x] += p << (2 * j);
    }
    __syncthreads();

    // load value from shared memory and rearange to output - the read value is
    // reused per warp
    uint32_t out = 0;

    // bit mask: Thread 0 always first input_bit_depth bits, thread 1 always
    // second input_bit_depth bits, ...
    const uint32_t mask = ((1 << 2) - 1) << (2 * (threadIdx.x % NPACK));

    #pragma unroll
    for (uint32_t j = 0; j < NPACK; j++) {
      uint32_t v = tmp[(threadIdx.x / NPACK) * NPACK + j] & mask;
      // retrieve correct bits
      v = v >> (2 * (threadIdx.x % NPACK));
      v = v << (2 * j);
      out += v;
    }

    size_t oidx = threadIdx.x / NPACK +
                  (threadIdx.x % NPACK) * (blockDim.x / NPACK) +
                  (i - threadIdx.x) / NPACK;
    output[oidx] = out;
    __syncthreads();
  }
}


} //namespace edd
} //namespace effelsberg
} //namespace psrdada_cpp
