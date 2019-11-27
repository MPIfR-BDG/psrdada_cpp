
#ifndef PSRDADA_CPP_SIGPROCHEADER_HPP
#define PSRDADA_CPP_SIGPROCHEADER_HPP

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

namespace psrdada_cpp
{

    struct FilHead {

        FilHead()
        : rawfile("unset")
        , source("unset")
        , az(0.0)
        , dec(0.0)
        , fch1(0.0)
        , foff(0.0)
        , ra(0.0)
        , rdm(0.0)
        , tsamp(0.0)
        , tstart(0.0)
        , za(0.0)
        , datatype(0)
        , barycentric(0)
        , ibeam(0)
        , machineid(0)
        , nbeams(0)
        , nbits(0)
        , nchans(0)
        , nifs(0)
        , telescopeid(0)
        {}
        ~FilHead(){};

        std::string rawfile;
        std::string source;
        double az;                      // azimuth angle in deg
        double dec;                     // source declination
        double fch1;                    // frequency of the top channel in MHz
        double foff;                    // channel bandwidth in MHz
        double ra;                      // source right ascension
        double rdm;                     // reference DM
        double tsamp;                   // sampling time in seconds
        double tstart;                  // observation start time in MJD format
        double za;                      // zenith angle in deg

        uint32_t datatype;                  // data type ID
        uint32_t barycentric;                // barucentric flag
        uint32_t ibeam;                      // beam number
        uint32_t machineid;
        uint32_t nbeams;
        uint32_t nbits;
        uint32_t nchans;
        uint32_t nifs;
        uint32_t telescopeid;
    };

class SigprocHeader
{
public:
    SigprocHeader();
    ~SigprocHeader();
    std::size_t write_header(RawBytes& block, PsrDadaHeader ph);
    std::size_t write_header(char*& ptr, FilHead& header); //should be const on FilHead
    void read_header(std::istream &infile, FilHead &header);
    void read_header(RawBytes& block, FilHead &header);
    double hhmmss_to_double(std::string const& val);

private:
    /*
     * @brief write string to the header
     */
    void header_write(char*& ptr, std::string const& str);
    void header_write(char*& ptr, std::string const& str, std::string const& name);

    /*
     * @brief write a value to the stream
     */
    template<typename NumericT>
    void header_write(char*& ptr, std::string const& name, NumericT val);

};

} // namespace psrdada_cpp
#include "psrdada_cpp/detail/sigprocheader.cpp"
#endif //PSRDADA_CPP_SIGPROCHEADER_HPP
