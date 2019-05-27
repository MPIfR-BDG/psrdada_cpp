#ifndef PSRDADA_CPP_COMMON_HPP
#define PSRDADA_CPP_COMMON_HPP

#define BOOST_LOG_DYN_LINK 1
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>

#include <cstddef>
#include <stdexcept>
#include <memory>
#include <string>
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <sys/types.h>

BOOST_LOG_TRIVIAL(debug) << std::setprecision(15);

#endif //PSRDADA_CPP_COMMON_HPP