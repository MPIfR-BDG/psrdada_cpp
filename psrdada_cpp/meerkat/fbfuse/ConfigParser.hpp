#ifndef PSRDADA_CPP_MEERKAT_CONFIGPARSER_HPP
#define PSRDADA_CPP_MEERKAT_CONFIGPARSER_HPP

#include "psrdada_cpp/meerkat/fbfuse/Config.hpp"
#include <string>

void parse_xml_config(std::string config_file, Config& config);

#endif //PSRDADA_CPP_MEERKAT_CONFIGPARSER_HPP