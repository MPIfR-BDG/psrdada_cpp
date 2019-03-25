#ifndef PSRDADA_CPP_MEERKAT_TUSE_TESTFILEWRITERTEST_H
#define PSRDADA_CPP_MEERKAT_TUSE_TESTFILEWRITERTEST_H

#include "psrdada_cpp/meerkat/tuse/test_file_writer.hpp"
#include <gtest/gtest.h>

namespace psrdada_cpp {
namespace meerkat {
namespace tuse {
namespace test {

class TestFileWriterTest: public ::testing::Test
{
protected:
    void SetUp() override;
    void TearDown() override;

public:
    TestFileWriterTest();
    ~TestFileWriterTest();

};

} //namespace test
} //namespace tuse
} //namespace meerkat
} //namespace psrdada_cpp

#endif //PSRDADA_CPP_MEERKAT_TUSE_TESTFILEWRITER_H
