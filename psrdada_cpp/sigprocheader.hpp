#ifndef PSRDADA_CPP_SIGPROCHEADER_HPP
#define PSRDADA_CPP_SIGPROCHEADER_HPP
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
#include"psrdada_cpp/psrdadaheader.hpp"
#include"psrdada_cpp/raw_bytes.hpp"

/* @detail: A SigProc Header writer class. This class will parse values
 *          from a PSRDADA header object and write that out as a standard
 *          SigProc format. This is specific for PSRDADA stream.
 */

namespace psrdada_cpp{


class SigprocHeader
{
    public:
 
        SigprocHeader();
        ~SigprocHeader();
        void write_header(RawBytes& block,PsrDadaHeader ph);

     
    private:
        /*
         * @brief write string to the header
         */ 
        template<class String>
        void header_write(RawBytes& block, String str); 
 
        /*
         * @brief write a value to the stream
         */
        template<class String, typename NumericT>
        void header_write(RawBytes& block, String name, NumericT val); 

};

} // namespace psrdada_cpp
#include "psrdada_cpp/detail/sigprocheader.cpp"
#endif //PSRDADA_CPP_SIGPROCHEADER_HPP
