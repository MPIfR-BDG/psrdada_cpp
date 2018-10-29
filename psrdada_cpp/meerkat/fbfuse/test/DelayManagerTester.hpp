#ifndef PSRDADA_CPP_MEERKAT_FBFUSE_DELAYMANAGERTEST_CUH
#define PSRDADA_CPP_MEERKAT_FBFUSE_DELAYMANAGERTEST_CUH

#include <gtest/gtest.h>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace test {

class DelayManagerTester: public ::testing::Test
{
protected:
    void SetUp() override;
    void TearDown() override;

public:
    DelayManagerTester();
    ~DelayManagerTester();
};

} //namespace test
} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

#endif //PSRDADA_CPP_MEERKAT_FBFUSE_DELAYMANAGERTEST_CUH