#include "psrdada_cpp/sigprocheader.hpp"
#include "psrdada_cpp/psrdadaheader.hpp"
#include <string>
#include <chrono>
#include <boost/algorithm/string.hpp>


namespace psrdada_cpp
{
    SigprocHeader::SigprocHeader()
    {
    }
    
    SigprocHeader::~SigprocHeader()
    {
    }

    void SigprocHeader::write_header(RawBytes& block, PsrDadaHeader ph)
    {
        header_write<std::string>(block,"HEADER_WRITE");
        header_write<std::string,std::uint32_t>(block,"telescope_id",0);
        header_write<std::string,std::uint32_t>(block,"machine_id",11);
        header_write<std::string,std::uint32_t>(block,"data_type",1);
        header_write<std::string,std::uint32_t>(block,"barycentric",0);
        header_write<std::string>(block,ph.source_name());
        // RA DEC 
        auto ra_val = ph.ra();
        auto dec_val =ph.dec();
        std::vector<std::string> ra_s;
        std::vector<std::string> dec_s;
        boost::split(ra_s,ra_val,boost::is_any_of(":"));
        boost::split(dec_s,dec_val,boost::is_any_of(":"));
        auto ra = stof(boost::join(ra_s," "));
        auto dec = stof(boost::join(dec_s," "));
        header_write<std::string,float>(block,"src_raj",ra);
        header_write<std::string,float>(block, "src_dej",dec);
        header_write<std::string,std::uint32_t>(block,"nbits",ph.nbits());
        header_write<std::string,std::uint32_t>(block,"nifs",1);
        header_write<std::string,std::uint32_t>(block,"nchans",ph.nchans());
        header_write<std::string,float>(block,"fch1", ph.freq());
        header_write<std::string,float>(block,"foff",ph.bw()/ph.nchans());
        header_write<std::string,float>(block,"tstart",ph.tstart());
        header_write<std::string,double>(block,"tsamp",ph.tsamp());
        header_write<std::string>(block,"HEADER_END");
    }

} // namespace psrdada_cpp

        
