#ifndef PSRDADA_CPP_PSRDADAHEADER_HPP
#define PSRDADA_CPP_PSRDADAHEADER_HPP

/*

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/


#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdlib.h>
#include"psrdada_cpp/raw_bytes.hpp"

#define DADA_HDR_SIZE 4096L

/*
 * @ detail: PsrDada header storage class
 * Reads the header block of DADA buffer and stores
 * all the necessary metadata to private members
 * which can be accessed via getters
 */

namespace psrdada_cpp
{
class PsrDadaHeader
{
public:

    PsrDadaHeader();
    ~PsrDadaHeader();

    /**
     * @brief: Get values from given key words
     */
    void from_bytes(RawBytes& block, std::uint32_t beamnum);

    /**
     * @ All getters for psrdada header
     */
    double bw() const;

    double freq() const;

    std::uint32_t nbits() const;

    double tsamp() const;

    std::string ra() const;

    std::string dec() const;

    std::string telescope() const;

    std::string instrument() const;

    std::string source_name() const;

    double tstart() const;

    std::uint32_t nchans() const;

    std::uint32_t beam() const;

    /**
     * @brief: All the setters
     */
    void set_bw(double bw);

    void set_freq(double freq);

    void set_nbits(std::uint32_t nbits);

    void set_tsamp(double tsamp);

    void set_beam(std::uint32_t beam);

    void set_ra(std::string ra);

    void set_dec(std::string dec);

    void set_telescope(std::string telescope);

    void set_instrument(std::string instrument);

    void set_source(std::string source);

    void set_tstart(double tstart);

    void set_nchans(std::uint32_t nchans);

private:
    std::string get_value(std::string name, std::stringstream& header) const;

    /**
     * @brief All standard PSRDADA header parameters (can add/subtract
     * if needed)
     */
    double _bw;
    double _freq;
    std::uint32_t _nchans;
    std::uint32_t _ndim;
    std::uint32_t _npol;
    std::uint32_t _nbits;
    double _tsamp;
    std::uint32_t _beam;
    std::string _source_name;
    std::string _ra;
    std::string _dec;
    std::string _telescope;
    std::string _instrument;
    double _mjd;

};
} // namespace psrdada_cpp

#endif // PSRDADA_CPP_PSRDADAHEADER_HPP
