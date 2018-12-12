#ifndef PSRDADA_CPP_EFFELSBERG_EDD_CHANNELISERTESTER_CUH
#define PSRDADA_CPP_EFFELSBERG_EDD_CHANNELISERTESTER_CUH

#include "psrdada_cpp/effelsberg/edd/Channeliser.cuh"
#include "psrdada_cpp/dada_db.hpp"
#include "thrust/host_vector.h"
#include <gtest/gtest.h>
#include <vector>

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {
namespace test {

class ChanneliserTester: public ::testing::Test
{
protected:
    void SetUp() override;
    void TearDown() override;

public:
    ChanneliserTester();
    ~ChanneliserTester();
    void performance_test(std::size_t nchans, std::size_t nsamps_per_packet, std::size_t nbits);
};

} //namespace test
} //namespace edd
} //namespace meerkat
} //namespace psrdada_cpp

#endif //PSRDADA_CPP_EFFELSBERG_EDD_CHANNELISERTESTER_CUH
