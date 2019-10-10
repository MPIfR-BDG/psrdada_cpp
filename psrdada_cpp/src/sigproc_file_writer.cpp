#include "psrdada_cpp/sigproc_file_writer.hpp"
#include <iomanip>
#include <ctime>

namespace psrdada_cpp {

SigprocFileWriter::SigprocFileWriter()
: _max_filesize(1<<20)
, _state(DISABLED)
, _tag("")
, _outdir("./")
, _extension(".fil")
, _total_bytes(0)
, _new_stream_required(false)
, _stream(nullptr)
{

}

SigprocFileWriter::~SigprocFileWriter()
{
    _stream.reset(nullptr);
}

void SigprocFileWriter::tag(std::string const& tag_)
{
    _tag = tag_;
}

std::string const& SigprocFileWriter::tag() const
{
    return _tag;
}

void SigprocFileWriter::directory(std::string const& dir)
{
    _outdir = dir;
}

std::string const& SigprocFileWriter::directory() const
{
    return _outdir;
}

void SigprocFileWriter::max_filesize(std::size_t size)
{
    _max_filesize = size;
}

std::size_t SigprocFileWriter::max_filesize() const
{
    return _max_filesize;
}

void SigprocFileWriter::init(RawBytes& block)
{
    SigprocHeader parser;
    parser.read_header(block, _header);
}

bool SigprocFileWriter::operator()(RawBytes& block)
{
    if (_state == ENABLED)
    {
        if ((_stream == nullptr) || _new_stream_required)
        {
            new_stream();
        }
        _stream->write(block.ptr(), block.used_bytes());
    }
    else if (_state == DISABLED)
    {
        if (_stream != nullptr)
        {
            _stream.reset(nullptr);
        }
    }
    _total_bytes += block.used_bytes();
    return false;
}

void SigprocFileWriter::new_stream()
{
    // Here we should update the tstart of the default header to be the
    // start of the stream
    _header.tstart = _header.tstart + ((
        (_total_bytes/(_header.nbits/8.0))/(_header.nchans)
        ) * _header.tsamp)/(86400.0);
    // reset the total bytes counter to keep the time tracked correctly
    _total_bytes = 0;
    //Generate the new base filename in <utc>_<tag> format
    std::stringstream base_filename;
    // Get UTC time string
    std::time_t unix_time = static_cast<std::time_t>((_header.tstart - 40587.0) * 86400.0);
    struct std::tm * ptm = std::gmtime(&unix_time);

    // Swapped out put_time call for strftime due to put_time
    // causing compiler bugs prior to g++ 5.x
    char formatted_time[80];
    strftime (formatted_time, 80, "%Y-%m-%d-%H:%M:%S", ptm);
    base_filename << formatted_time;

    if (_tag != "")
    {
        base_filename << "_" << _tag;
    }

    _stream.reset(
        new FileStream(
            _outdir,
            base_filename.str(),
            _extension,
            _max_filesize,
            [&](std::size_t& header_size, std::size_t bytes_written)
            -> std::shared_ptr<char const>
            {
                // We do not explicitly delete[] this array
                // Cleanup is handled by the shared pointer
                // created below
                char* temp_header = new char[4096];
                SigprocHeader parser;
                // Make a copy of the header to edit
                FilHead fh = header();
                //Here we do any required updates to the header
                fh.tstart = fh.tstart + ((
                    (bytes_written/(fh.nbits/8.0))/(fh.nchans)
                    ) * fh.tsamp)/(86400.0);
                header_size = parser.write_header(temp_header, fh);
                std::shared_ptr<char const> header_ptr(
                    temp_header, std::default_delete<char[]>());
                return header_ptr;
            }
            ));
    _new_stream_required = false;
}

void SigprocFileWriter::enable()
{
    if (_state == DISABLED)
    {
        _new_stream_required = true;
    }
    _state = ENABLED;
}

void SigprocFileWriter::disable()
{
    _state = DISABLED;
    _new_stream_required = false;
}

FilHead& SigprocFileWriter::header()
{
    return _header;
}

}//psrdada_cpp


