#ifndef PSRDADA_CPP_TEST_SIGPROC_FILE_WRITER_HPP
#define PSRDADA_CPP_TEST_SIGPROC_FILE_WRITER_HPP

#include <gtest/gtest.h>

namespace psrdada_cpp {
namespace test {

class TestSigprocFileWriter: public ::testing::Test
{
protected:
    void SetUp() override;
    void TearDown() override;

public:
    TestSigprocFileWriter();
    ~TestSigprocFileWriter();

};

} //namespace test
} //namespace psrdada_cpp

#endif //PSRDADA_CPP_SIGPROC_FILE_WRITER_HPP
