#include "psrdada_cpp/sigprocheader.hpp"
#include "psrdada_cpp/psrdadaheader.hpp"
#include <string>
#include <chrono>
#include <boost/algorithm/string.hpp>
#include <iterator>

namespace psrdada_cpp
{

    SigprocHeader::SigprocHeader()
    :
    _header_size(0)
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
        header_write<double>(ptr,"fch1", ph.freq() + ph.bw()/2.0);
        header_write<double>(ptr,"foff",-1.0*abs(ph.bw()/ph.nchans()));
        header_write<double>(ptr,"tstart",ph.tstart());
        header_write<double>(ptr,"tsamp",ph.tsamp());
        header_write(ptr,"HEADER_END");
        _header_size = std::distance(block.ptr(),ptr);
    }

    std::size_t SigprocHeader::header_size() const
    {
	    return _header_size;
    }

    void SigprocHeader::read_header(std::ifstream &infile, FilHead &header) {

        std::string read_param;
	    char field[60];

	    int fieldlength;

        while(true) {
            infile.read((char *)&fieldlength, sizeof(int));
            infile.read(field, fieldlength * sizeof(char));
            field[fieldlength] = '\0';
            read_param = field;

            if (read_param == "HEADER_END") break;		// finish reading the header when its end is reached
            else if (read_param == "rawdatafile") {
                infile.read((char *)&fieldlength, sizeof(int));		// reads the length of the raw data file name
                infile.read(field, fieldlength * sizeof(char));
                field[fieldlength] = '\0';
                header.rawfile = field;
            }
            else if (read_param == "source_name") {
                infile.read((char *)&fieldlength, sizeof(int));
                infile.read(field, fieldlength * sizeof(char));
                field[fieldlength] = '\0';
                header.source = field;
            }
            else if (read_param == "machine_id")	infile.read((char *)&header.machineid, sizeof(int));
            else if (read_param == "telescope_id")	infile.read((char *)&header.telescopeid, sizeof(int));
            else if (read_param == "src_raj")	infile.read((char *)&header.ra, sizeof(double));
            else if (read_param == "src_dej")	infile.read((char *)&header.dec, sizeof(double));
            else if (read_param == "az_start")	infile.read((char *)&header.az, sizeof(double));
            else if (read_param == "za_start")	infile.read((char *)&header.za, sizeof(double));
            else if (read_param == "data_type")	infile.read((char *)&header.datatype, sizeof(int));
            else if (read_param == "refdm")		infile.read((char *)&header.rdm, sizeof(double));
            else if (read_param == "nchans")	infile.read((char *)&header.nchans, sizeof(int));
            else if (read_param == "fch1")		infile.read((char *)&header.fch1, sizeof(double));
            else if (read_param == "foff")		infile.read((char *)&header.foff, sizeof(double));
            else if (read_param == "nbeams")	infile.read((char *)&header.nbeams, sizeof(int));
            else if (read_param == "ibeam")		infile.read((char *)&header.ibeam, sizeof(int));
            else if (read_param == "nbits")		infile.read((char *)&header.nbits, sizeof(int));
            else if (read_param == "tstart")	infile.read((char *)&header.tstart, sizeof(double));
            else if (read_param == "tsamp")		infile.read((char *)&header.tsamp, sizeof(double));
            else if (read_param == "nifs")		infile.read((char *)&header.nifs, sizeof(int));
        }
    }

} // namespace psrdada_cpp
