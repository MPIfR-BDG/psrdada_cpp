#include"psrdada_cpp/psrdadaheader.hpp"
#include <iostream>
#include <chrono>



namespace psrdada_cpp{

PsrDadaHeader::PsrDadaHeader()
{
}

PsrDadaHeader::~PsrDadaHeader()
{
}

void PsrDadaHeader::from_bytes(RawBytes& block, std::uint32_t beamnum)
{
    std::vector<char> buf(DADA_HDR_SIZE);
    std::copy(block.ptr(),block.ptr()+block.total_bytes(),buf.begin());
    std::stringstream header;
    header.rdbuf()->pubsetbuf(&buf[0],DADA_HDR_SIZE);
    set_bw(atof(get_value("BW ",header).c_str()));
    set_freq(atof(get_value("FREQ ",header).c_str()));
    set_nchans(atoi(get_value("NCHAN ",header).c_str()));
    set_nbits(atoi(get_value("NBIT ",header).c_str()));
    set_tsamp(atof(get_value("TSAMP ",header).c_str()));
    set_beam(atoi(get_value("IBEAM" + std::to_string(beamnum) + " ", header).c_str()));
    set_source(get_value("SOURCE" + std::to_string(beamnum) + " ",header));
    set_ra(get_value("RA" + std::to_string(beamnum) + " ",header));
    set_dec(get_value("DEC" + std::to_string(beamnum) + " ",header));
    set_telescope(get_value("TELESCOPE ",header));
    set_instrument(get_value("INSTRUMENT ",header));
    // Getting the correct MJD
    double sync_mjd = atof(get_value("SYNC_TIME_MJD", header).c_str());
    double sample_clock = atof(get_value("SAMPLE_CLOCK", header).c_str());
    double sample_clock_start = atof(get_value("SAMPLE_CLOCK_START", header).c_str());
    set_tstart(sync_mjd + (double)(sample_clock_start/sample_clock/86400.0));
    return;
}

std::string PsrDadaHeader::get_value(std::string name,std::stringstream& header)
{
    size_t position = header.str().find(name);
    if (position!=std::string::npos)
    {
        header.seekg(position+name.length());
        std::string value;
        header >> value;
        return value;
    } 
    else 
    {
      return "";
    }
}

double PsrDadaHeader::bw()
{
    return _bw;
}

double PsrDadaHeader::freq()
{
    return _freq;
}

std::uint32_t PsrDadaHeader::nbits()
{
    return _nbits;
}

double PsrDadaHeader::tsamp()
{
    return _tsamp;
}

std::uint32_t PsrDadaHeader::beam()
{
    return _beam;
}

std::string PsrDadaHeader::ra() 
{
    return _ra;
}

std::string PsrDadaHeader::dec() 
{
    return _dec;
}

std::string PsrDadaHeader::telescope()
{
    return _telescope;
}

std::string PsrDadaHeader::instrument()
{
    return _instrument;
}

std::string PsrDadaHeader::source_name()
{
    return _source_name;
}

std::uint32_t PsrDadaHeader::nchans()
{
    return _nchans;
}

double PsrDadaHeader::tstart()
{
    return _mjd;
}

void PsrDadaHeader::set_bw(double bw)
{
    _bw = bw;
}

void PsrDadaHeader::set_freq(double freq)
{
    _freq=freq;
}

void PsrDadaHeader::set_nbits(std::uint32_t nbits)
{
    _nbits=nbits;
}

void PsrDadaHeader::set_tsamp(double tsamp)
{
    _tsamp = tsamp;
}

void PsrDadaHeader::set_beam(std::uint32_t beam)
{
    _beam = beam;
}

void PsrDadaHeader::set_ra(std::string ra)
{
    _ra.assign(ra);
}

void PsrDadaHeader::set_dec(std::string dec)
{
    _dec.assign(dec);
}

void PsrDadaHeader::set_telescope(std::string telescope)
{
    _telescope = telescope;
}

void PsrDadaHeader::set_instrument(std::string instrument)
{
    _instrument = instrument;
}

void PsrDadaHeader::set_source(std::string source)
{
    _source_name=source;
}

void PsrDadaHeader::set_tstart(double tstart)
{
    _mjd = tstart;
}

void PsrDadaHeader::set_nchans(std::uint32_t nchans)
{
    _nchans=nchans;
}

} // namespace psrdada_cpp
