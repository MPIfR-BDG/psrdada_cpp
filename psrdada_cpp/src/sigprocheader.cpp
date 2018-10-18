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

    void SigprocHeader::header_write(char*& ptr, std::string const& str)
    {
        int len = str.size();
        std::memcpy(ptr,(char*)&len,sizeof(len));
	ptr += sizeof(len);
        std::copy(str.begin(),str.end(),ptr);
	ptr += len;
    }

     void SigprocHeader::header_write(char*& ptr, std::string const& str, std::string const& name)
    {
	header_write(ptr,str);
	header_write(ptr,name);
    }

    void SigprocHeader::write_header(RawBytes& block, PsrDadaHeader ph)
    {
	auto ptr = block.ptr();
        header_write(ptr,"HEADER_START");
        header_write<std::uint32_t>(ptr,"telescope_id",0);
        header_write<std::uint32_t>(ptr,"machine_id",11);
        header_write<std::uint32_t>(ptr,"data_type",1);
        header_write<std::uint32_t>(ptr,"barycentric",0);
        header_write(ptr,"source_name",ph.source_name());
        // RA DEC 
        auto ra_val = ph.ra();
        auto dec_val =ph.dec();
	std::vector<std::string> ra_s;
	std::vector<std::string> dec_s;
        boost::split(ra_s,ra_val,boost::is_any_of(":"));
        boost::split(dec_s,dec_val,boost::is_any_of(":"));
        double ra = stod(boost::join(ra_s,""));
        double dec = stod(boost::join(dec_s,""));
        header_write<double>(ptr,"src_raj",ra);
        header_write<double>(ptr, "src_dej",dec);
        header_write<std::uint32_t>(ptr,"nbits",ph.nbits());
        header_write<std::uint32_t>(ptr,"nifs",1);
        header_write<std::uint32_t>(ptr,"nchans",ph.nchans());
        header_write<double>(ptr,"fch1", ph.freq());
        header_write<double>(ptr,"foff",ph.bw()/ph.nchans());
        header_write<double>(ptr,"tstart",ph.tstart());
        header_write<double>(ptr,"tsamp",ph.tsamp());
        header_write(ptr,"HEADER_END");
    }


} // namespace psrdada_cpp

        
