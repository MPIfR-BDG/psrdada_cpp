#include "gtest/gtest.h"
#include "psrdada_cpp/cli_utils.hpp"
#include <cstdlib>

int main(int argc, char **argv) {
  char * val = getenv("LOG_LEVEL");
  if ( val )
  {
      psrdada_cpp::set_log_level(val);
  }

  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}

