#ifndef PSRDADA_CPP_EFFELSBERG_EDD_EDDPOLNMERGETESTER_HPP
#define PSRDADA_CPP_EFFELSBERG_EDD_EDDPOLNMERGETESTER_HPP

#include <gtest/gtest.h>

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {
namespace test {

class EDDPolnMergeTester: public ::testing::Test
{
protected:
    void SetUp() override;
    void TearDown() override;

public:
    EDDPolnMergeTester();
    ~EDDPolnMergeTester();

};

} //namespace test
} //namespace edd
} //namespace effelsberg
} //namespace psrdada_cpp

#endif //EFFELSBERG_EDD_EDDPOLNMERGETESTER_HPP
