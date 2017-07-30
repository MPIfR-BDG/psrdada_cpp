#ifndef PSRDADA_CPP_CLI_UTILS_HPP
#define PSRDADA_CPP_CLI_UTILS_HPP

#include "psrdada_cpp/common.hpp"

namespace psrdada_cpp {

    key_t string_to_key(std::string const& in);
    void set_log_level(std::string level);

} //namespace
#endif //PSRDADA_CPP_CLI_UTILS_HPP