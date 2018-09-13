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
	auto ptr = block.ptr();
        header_write<std::string>(ptr,"HEADER_START");
        header_write<std::string,std::uint32_t>(ptr,"telescope_id",0);
        header_write<std::string,std::uint32_t>(ptr,"machine_id",11);
        header_write<std::string,std::uint32_t>(ptr,"data_type",1);
        header_write<std::string,std::uint32_t>(ptr,"barycentric",0);
        header_write<std::string>(ptr,ph.source_name());
        // RA DEC 
        auto ra_val = ph.ra();
        auto dec_val =ph.dec();
        std::vector<std::string> ra_s;
        std::vector<std::string> dec_s;
        boost::split(ra_s,ra_val,boost::is_any_of(":"));
        boost::split(dec_s,dec_val,boost::is_any_of(":"));
        auto ra = stof(boost::join(ra_s," "));
        auto dec = stof(boost::join(dec_s," "));
        header_write<std::string,float>(ptr,"src_raj",ra);
        header_write<std::string,float>(ptr, "src_dej",dec);
        header_write<std::string,std::uint32_t>(ptr,"nbits",ph.nbits());
        header_write<std::string,std::uint32_t>(ptr,"nifs",1);
        header_write<std::string,std::uint32_t>(ptr,"nchans",ph.nchans());
        header_write<std::string,float>(ptr,"fch1", ph.freq());
        header_write<std::string,float>(ptr,"foff",ph.bw()/ph.nchans());
        header_write<std::string,float>(ptr,"tstart",ph.tstart());
        header_write<std::string,double>(ptr,"tsamp",ph.tsamp());
        header_write<std::string>(ptr,"HEADER_END");
    }


} // namespace psrdada_cpp

        
