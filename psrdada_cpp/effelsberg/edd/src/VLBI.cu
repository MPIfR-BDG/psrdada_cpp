#include "psrdada_cpp/effelsberg/edd/VLBI.cuh"

#include "psrdada_cpp/cuda_utils.hpp"

#define EDD_NTHREADS_PACK 1024 
#define NPACK 16

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {

// Create  abit mask with 1 between first and lastBit (inclusive) and zero
/// otherwise;
uint32_t bitMask(uint32_t firstBit, uint32_t lastBit)
{
   uint32_t mask = 0U;
   for (uint32_t i=firstBit; i<=lastBit; i++)
       mask |= 1 << i;
   return mask;
}

/// Squeeze a value into the specified bitrange of the target
void setBitsWithValue(uint32_t &target, uint32_t firstBit, uint32_t lastBit, uint32_t value)
{
	// check if value is larger than bit range
	if (value > (1 << (lastBit + 1 - firstBit)))
	{
				std::cerr << "value: " << value << ", 1 << (last-bit - firstbit) " << (1 << (lastBit - firstBit)) << ", bitrange: " << lastBit-firstBit << std::endl;
				throw std::runtime_error("Value does not fit into bitrange");
	}

	uint32_t mask = bitMask(firstBit, lastBit);

	// zero out relevant bits in data
	target &= ~mask;

	// shift value to corerct position
	value = value << firstBit;

	// update target with value
	target |= value;
}

/// get numerical value from the specified bits in the target 
uint32_t getBitsValue(const uint32_t &target, uint32_t firstBit, uint32_t lastBit)
{
	uint32_t mask = bitMask(firstBit, lastBit);

	uint32_t res = target & mask;

	return res >> firstBit;
}


VDIFHeader::VDIFHeader()
{
  for (int i=0; i < 8; i++)
  {
    data[i] = 0U;
  }

  // set standard VDIF header
  setBitsWithValue(data[1], 30, 30, 0);
  setBitsWithValue(data[1], 30, 31, 0);

  // set Version Number to 1
  setBitsWithValue(data[2], 29, 31, 1);
}

uint32_t* VDIFHeader::getData()
{
  return data;
}

void VDIFHeader::setInvalid()
{
  setBitsWithValue(data[0], 31, 31, 1);
}

void VDIFHeader::setValid()
{
  setBitsWithValue(data[0], 31, 31, 0);
}

bool VDIFHeader::isValid() const
{
  return (getBitsValue(data[0], 31, 31) == 0);
}

void VDIFHeader::setSecondsFromReferenceEpoch(uint32_t value)
{
  setBitsWithValue(data[0], 0, 29, value);
}

uint32_t VDIFHeader::getSecondsFromReferenceEpoch() const
{
  return getBitsValue(data[0], 0, 29);
}

void VDIFHeader::setReferenceEpoch(uint32_t value)
{
  setBitsWithValue(data[1], 24, 29, value);
}

uint32_t VDIFHeader::getReferenceEpoch() const
{
  return getBitsValue(data[1], 24, 29);
}

void VDIFHeader::setDataFrameNumber(uint32_t value)
{
  setBitsWithValue(data[1], 0, 23, value);
}

uint32_t VDIFHeader::getDataFrameNumber() const
{
  return getBitsValue(data[1], 0, 23);
}

void VDIFHeader::setDataFrameLength(uint32_t value)
{
  setBitsWithValue(data[2], 0, 23, value);
}

uint32_t VDIFHeader::getDataFrameLength() const
{
  return getBitsValue(data[2], 0, 23);
}

uint32_t VDIFHeader::getVersionNumber() const
{
  return getBitsValue(data[2], 29, 31);
}

void VDIFHeader::setNumberOfChannels(uint32_t value)
{
  setBitsWithValue(data[2], 24, 28, value);
}

uint32_t VDIFHeader::getNumberOfChannels() const
{
  return getBitsValue(data[2], 24, 28);
}

bool VDIFHeader::isRealDataType() const
{
  return (getBitsValue(data[3], 31, 31) == 0);
}

bool VDIFHeader::isComplexDataType() const
{
  return (getBitsValue(data[3], 31, 31) == 1);
}

void VDIFHeader::setComplexDataType()
{
  setBitsWithValue(data[3], 31, 31, 1);
}

void VDIFHeader::setRealDataType()
{
  setBitsWithValue(data[0], 31, 31, 0);
}

void VDIFHeader::setBitsPerSample(uint32_t value)
{
  setBitsWithValue(data[3], 26, 30, value);
}

uint32_t VDIFHeader::getBitsPerSample() const
{
  return getBitsValue(data[3], 26, 30);
}

void VDIFHeader::setThreadId(uint32_t value)
{
  setBitsWithValue(data[3], 16, 25, value);
}

uint32_t VDIFHeader::getThreadId() const
{
  return getBitsValue(data[3], 16, 25);
}

void VDIFHeader::setStationId(uint32_t value)
{
  setBitsWithValue(data[3], 0, 15, value);
}

uint32_t VDIFHeader::getStationId() const
{
  return getBitsValue(data[3], 0, 15);
}





} //namespace edd
} //namespace effelsberg
} //namespace psrdada_cpp
