#include "psrdada_cpp/meerkat/fbfuse/test/DelayManagerTester.hpp"

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace test {

class DelayManagerTester: public ::testing::Test
{
protected:
    void setUp() override;
    void tearDown() override;

public:
    DelayManagerTester();
    ~DelayManagerTester();
};


DelayManagerTester::DelayManagerTester()
    : ::testing::Test()
{

}

DelayManagerTester::~DelayManagerTester()
{

}

void DelayManagerTester::SetUp()
{

}

void DelayManagerTester::TearDown()
{

}

TEST_F(DelayManagerTester, dummy_test)
{
    ASSERT_TRUE(1 == 1);
}

} //namespace test
} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

