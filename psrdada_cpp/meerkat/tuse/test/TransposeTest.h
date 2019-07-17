#ifndef PSRDADA_CPP_MEERKAT_TUSE_TRANSPOSETEST_H
#define PSRDADA_CPP_MEERKAT_TUSE_TRANSPOSETEST_H

#include "psrdada_cpp/meerkat/tuse/transpose_to_dada.hpp"
#include <gtest/gtest.h>

namespace psrdada_cpp {
namespace meerkat {
namespace tuse {
namespace test {

class TransposeTest: public ::testing::Test
{
protected:
    void SetUp() override;
    void TearDown() override;

public:
    TransposeTest();
    ~TransposeTest();

};

} //namespace test
} //namespace tuse
} //namespace meerkat
} //namespace psrdada_cpp

#endif //PSRDADA_CPP_MEERKAT_TUSE_TRANSPOSETEST_H
