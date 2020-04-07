#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/effelsberg/edd/test/SKTestVector.hpp"
#include "psrdada_cpp/effelsberg/edd/SpectralKurtosis.hpp"
#include <gtest/gtest.h>

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {
namespace test {

class SpectralKurtosisTester: public ::testing::Test
{
public:
    SpectralKurtosisTester();
    ~SpectralKurtosisTester();

protected:
    void SetUp() override;
    void TearDown() override;
};

} //test
} //edd
} //effelsberg
} //psrdada_cpp




