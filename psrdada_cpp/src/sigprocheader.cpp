#include "psrdada_cpp/sigprocheader.hpp"
#include "psrdada_cpp/psrdadaheader.hpp"
#include <string>
#include <chrono>
#include <boost/algorithm/string.hpp>
#include <iterator>

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

    std::size_t SigprocHeader::write_header(RawBytes& block, PsrDadaHeader ph)
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
        auto dec_val = ph.dec();
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
        header_write<std::uint32_t>(ptr, "ibeam", ph.beam());
        header_write<double>(ptr,"fch1", ph.freq() + (ph.bw()/2.0) + (ph.bw()/(double)ph.nchans())/2.0);
        header_write<double>(ptr,"foff",(-1.0 * ph.bw()/(double)ph.nchans()));
        header_write<double>(ptr,"tstart",ph.tstart());
        header_write<double>(ptr,"tsamp",ph.tsamp());
        header_write(ptr,"HEADER_END");
        return std::distance(block.ptr(),ptr);
    }

    std::size_t SigprocHeader::write_header(char*& ptr, FilHead& ph)
    {
	    char* new_ptr = ptr;
        header_write(new_ptr,"HEADER_START");
        header_write<std::uint32_t>(new_ptr,"telescope_id",ph.telescopeid);
        header_write<std::uint32_t>(new_ptr,"machine_id",ph.machineid);
        header_write<std::uint32_t>(new_ptr,"data_type",ph.datatype);
        header_write<std::uint32_t>(new_ptr,"barycentric",ph.barycentric);
        header_write(new_ptr,"source_name",ph.source);
        header_write<double>(new_ptr,"src_raj",ph.ra);
        header_write<double>(new_ptr, "src_dej",ph.dec);
        header_write<std::uint32_t>(new_ptr,"nbits",ph.nbits);
        header_write<std::uint32_t>(new_ptr,"nifs",ph.nifs);
        header_write<std::uint32_t>(new_ptr,"nchans",ph.nchans);
        header_write<std::uint32_t>(new_ptr, "ibeam", ph.ibeam);
        header_write<double>(new_ptr,"fch1", ph.fch1);
        header_write<double>(new_ptr,"foff",ph.foff);
        header_write<double>(new_ptr,"tstart",ph.tstart);
        header_write<double>(new_ptr,"tsamp",ph.tsamp);
        header_write(new_ptr,"HEADER_END");
	    return (std::size_t) (new_ptr - ptr);
    }

    void SigprocHeader::read_header(std::ifstream &infile, FilHead &header)
    {

        std::string read_param;
	    char field[60];

	    int fieldlength=0;

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
            else if (read_param == "machine_id")	infile.read((char *)&header.machineid, sizeof(uint32_t));
            else if (read_param == "telescope_id")	infile.read((char *)&header.telescopeid, sizeof(uint32_t));
            else if (read_param == "src_raj")	infile.read((char *)&header.ra, sizeof(double));
            else if (read_param == "src_dej")	infile.read((char *)&header.dec, sizeof(double));
            else if (read_param == "az_start")	infile.read((char *)&header.az, sizeof(double));
            else if (read_param == "za_start")	infile.read((char *)&header.za, sizeof(double));
            else if (read_param == "data_type")	infile.read((char *)&header.datatype, sizeof(uint32_t));
            else if (read_param == "barycentric")	infile.read((char *)&header.barycentric, sizeof(uint32_t));
            else if (read_param == "refdm")		infile.read((char *)&header.rdm, sizeof(double));
            else if (read_param == "nchans")	infile.read((char *)&header.nchans, sizeof(uint32_t));
            else if (read_param == "fch1")		infile.read((char *)&header.fch1, sizeof(double));
            else if (read_param == "foff")		infile.read((char *)&header.foff, sizeof(double));
            else if (read_param == "nbeams")	infile.read((char *)&header.nbeams, sizeof(uint32_t));
            else if (read_param == "ibeam")		infile.read((char *)&header.ibeam, sizeof(uint32_t));
            else if (read_param == "nbits")		infile.read((char *)&header.nbits, sizeof(uint32_t));
            else if (read_param == "tstart")	infile.read((char *)&header.tstart, sizeof(double));
            else if (read_param == "tsamp")		infile.read((char *)&header.tsamp, sizeof(double));
            else if (read_param == "nifs")		infile.read((char *)&header.nifs, sizeof(uint32_t));
        }
    }

    void SigprocHeader::read_header(std::stringstream &infile, FilHead &header)
    {

        std::string read_param;
	    char field[60];

	    std::int32_t fieldlength = 0;

        while(true) {
            infile.read((char *)&fieldlength, sizeof(int));
            infile.read(field, fieldlength * sizeof(char));
            field[fieldlength] = '\0';
            read_param = field;

            if (read_param == "HEADER_END") break;		// finish reading the header when its end is reached
            else if (read_param == "rawdatafile") {
                infile.read((char *)&fieldlength, sizeof(int));
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
            else if (read_param == "barycentric")	infile.read((char *)&header.barycentric, sizeof(int));
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

    void SigprocHeader::read_header(RawBytes& block, FilHead& header)
    {
        std::stringstream instream;
        instream.write(block.ptr(), block.used_bytes());
        read_header(instream, header);
    }

    double SigprocHeader::hhmmss_to_double(std::string const& hhmmss_string)
    {
        try
        {
            std::stringstream stream(hhmmss_string);
            std::string hh, mm, ss;
            std::getline(stream, hh, ':');
            std::getline(stream, mm, ':');
            std::getline(stream, ss, ':');
            double hh_d = std::stod(hh);
            double mm_d = std::stod(mm);
            double ss_d = std::stod(ss);
            double val = 10000 * std::abs(hh_d) + 100 * mm_d + ss_d;
            if (hh_d < 0)
            {
                return -1 * val;
            }
            else
            {
                return val;
            }
        }
        catch (std::exception& e)
        {
	    std::stringstream error_msg;
            error_msg << "Error while converting " << hhmmss_string << " to sigproc format: " << e.what();
    	    BOOST_LOG_TRIVIAL(error) << error_msg.str();
            throw std::runtime_error(error_msg.str());
        }

    }

} // namespace psrdada_cpp
