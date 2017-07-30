#include "psrdada_cpp/cli_utils.hpp"

namespace psrdada_cpp {
    key_t string_to_key(std::string const& in)
    {
        key_t key;
        std::stringstream converter;
        converter << std::hex << in;
        converter >> key;
        return key;
    }

    void set_log_level(std::string level)
    {
        using namespace boost::log;

        if (level == "debug")
        {
            std::cout << "debug" << std::endl;
            core::get()->set_filter(trivial::severity >= trivial::debug);
        }
        else if (level == "info")
        {
            std::cout << "info" << std::endl;
            core::get()->set_filter(trivial::severity >= trivial::info);
        }
        else if (level == "warning")
        {
            std::cout << "warning" << std::endl;
            core::get()->set_filter(trivial::severity >= trivial::warning);
        }
        else
        {
            std::cout << "error" << std::endl;
            core::get()->set_filter(trivial::severity >= trivial::error);
        }
    }
} //namespace