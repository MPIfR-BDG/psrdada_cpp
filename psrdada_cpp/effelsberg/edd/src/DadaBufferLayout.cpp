#include "psrdada_cpp/effelsberg/edd/DadaBufferLayout.hpp"

#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/dada_client_base.hpp"

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {

DadaBufferLayout::DadaBufferLayout() {};

DadaBufferLayout::DadaBufferLayout(key_t input_key, size_t heapSize, size_t nSideChannels)
{
  intitialize(input_key, heapSize, nSideChannels);
}

void DadaBufferLayout::intitialize(key_t input_key, size_t heapSize, size_t nSideChannels)
{
    _input_key = input_key;
    _heapSize = heapSize;
    _nSideChannels = nSideChannels;

  MultiLog log("DadaBufferLayout");
  DadaClientBase client(input_key, log);
  _bufferSize = client.data_buffer_size();

  _sideChannelSize = nSideChannels * sizeof(int64_t);
   size_t totalHeapSize = _heapSize + _sideChannelSize;
  _nHeaps = _bufferSize / totalHeapSize;
  _gapSize = (_bufferSize - _nHeaps * totalHeapSize);
  _dataBlockBytes = _nHeaps * _heapSize;

  BOOST_LOG_TRIVIAL(debug) << "Memory configuration of dada buffer '" << _input_key << "' with " << nSideChannels << " side channels items and heapsize " << heapSize << " byte: \n"
                           << "  total size of buffer: " << _bufferSize << " byte\n"
                           << "  number of heaps per buffer: " << _nHeaps << "\n"
                           << "  datablock size in buffer: " << _dataBlockBytes << " byte\n"
                           << "  resulting gap: " << _gapSize << " byte\n"
                           << "  size of sidechannel data: " << _sideChannelSize << " byte\n";
}

key_t DadaBufferLayout::getInputkey() const
{
  return _input_key;
}


size_t DadaBufferLayout::getBufferSize() const
{
  return _bufferSize;
}

size_t DadaBufferLayout::getHeapSize() const
{
  return  _heapSize;
}

size_t DadaBufferLayout::getNSideChannels() const
{
  return _nSideChannels;
}

size_t DadaBufferLayout::sizeOfData() const
{
  return _dataBlockBytes;
}

size_t DadaBufferLayout::sizeOfGap() const
{
  return _gapSize;
}

size_t DadaBufferLayout::sizeOfSideChannelData() const
{
  return _sideChannelSize * _nHeaps;
}

size_t DadaBufferLayout::getNHeaps() const
{
  return _nHeaps;
}

} // edd
} // effelsberg
} // psrdada_cpp
