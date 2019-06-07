#ifndef PSRDADA_CPP_MEERKAT_FBFUSE_BUFFERDUMPTESTER_HPP
#define PSRDADA_CPP_MEERKAT_FBFUSE_BUFFERDUMPTESTER_HPP

#include "psrdada_cpp/meerkat/fbfuse/BufferDump.hpp"
#include <gtest/gtest.h>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace test {

class BufferDumpTester: public ::testing::Test
{
protected:
    void SetUp() override;
    void TearDown() override;

public:
    BufferDumpTester();
    ~BufferDumpTester();

protected:
};

} //namespace test
} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

#endif //PSRDADA_CPP_MEERKAT_FBFUSE_BUFFERDUMPTESTER_HPP
