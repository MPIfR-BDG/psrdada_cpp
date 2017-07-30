#ifndef PSRDADA_CPP_CLI_UTILS_HPP
#define PSRDADA_CPP_CLI_UTILS_HPP

#include "psrdada_cpp/common.hpp"

namespace psrdada_cpp {

    /**
     * @brief      Convert a string into a shared memory key
     *
     * @param      in    A hexadecimal string (e.g. "dada")
     *
     * @note       No error checking is performed on the conversion, meaning
     *             that non-hex valiues will still give an output (e.g. an input
     *             of "dag" will drop the non-hex "g" character giving an output
     *             of 0xda)
     *
     * @return     A key_t representation of the hexadecimal string
     */
    key_t string_to_key(std::string const& in);

    /**
     * @brief      Sets the log level for boost logging.
     *
     * @param[in]  level  The desired log level as a string
     *                    [debug, info, warning, error].
     */
    void set_log_level(std::string level);

} //namespace
#endif //PSRDADA_CPP_CLI_UTILS_HPP