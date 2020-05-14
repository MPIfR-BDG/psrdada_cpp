#include "dada_hdu.h"

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {


/**
 @class DadaBufferLayout
 @brief Calculate the data layout of the buffer
 */
class DadaBufferLayout
{
  private:
    size_t _bufferSize;
    key_t _input_key;
    size_t _heapSize;
    size_t _nSideChannels;

    size_t _sideChannelSize;
    size_t _nHeaps;
    size_t _gapSize;
    size_t _dataBlockBytes;


  public:
    // input key of the dadad buffer
    // size of spead heaps in bytes
    //  number of side channels
    DadaBufferLayout();
    DadaBufferLayout(key_t input_key , size_t heapSize, size_t nSideChannels);
    void intitialize(key_t input_key , size_t heapSize, size_t nSideChannels);

    key_t getInputkey() const;

    size_t getBufferSize() const;

    // get size of heaps in bytes
    size_t getHeapSize() const;

    size_t getNSideChannels() const;

    // returns size of data in buffer block in bytes
    size_t sizeOfData() const;

    // return size of gap between data and side channel
    size_t sizeOfGap() const;

    // returns size of side channelm data in buffer block in bytes
    size_t sizeOfSideChannelData() const;

    // number of heaps stored in one block of the buffer
    size_t getNHeaps() const;
};


} // edd
} // effelsberg
} // psrdada_cpp
