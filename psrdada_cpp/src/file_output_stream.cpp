#include "psrdada_cpp/file_output_stream.hpp"
#include "psrdada_cpp/common.hpp"
#include <iomanip>

namespace psrdada_cpp {

/*
 * @brief: A file writer for outputing SigProc format files.
 */

FileStream::File::File(std::string const& fname, std::size_t bytes)
: _full_path(fname)
, _bytes_requested(bytes)
, _bytes_written(0)
{
    _stream.open(_full_path, std::ifstream::out | std::ifstream::binary);
    if (_stream.is_open())
    {
        BOOST_LOG_TRIVIAL(info) << "Opened output file " << _full_path;
    }
    else
    {
        std::stringstream error_message;
        error_message << "Could not open file " << _full_path;
        BOOST_LOG_TRIVIAL(error) << error_message.str();
        throw std::runtime_error(error_message.str());
    }
}

FileStream::File::~File()
{
    if (_stream.is_open())
    {
        BOOST_LOG_TRIVIAL(info) << "Closing file " << _full_path;
        _stream.close();
    }
}

std::size_t FileStream::File::write(char const* ptr, std::size_t bytes)
{
    BOOST_LOG_TRIVIAL(debug) << "Writing " << bytes << " bytes to "<< _full_path;
    std::size_t bytes_remaining = _bytes_requested - _bytes_written;
    if (bytes > bytes_remaining)
    {
        _stream.write(ptr, bytes_remaining);
        _bytes_written += bytes_remaining;
        BOOST_LOG_TRIVIAL(debug) << "Partial write of " << bytes_remaining << " bytes";
        return bytes_remaining;
    }
    else
    {
        _stream.write(ptr, bytes);
        _bytes_written += bytes;
        BOOST_LOG_TRIVIAL(debug) << "Completed write";
        return bytes;
    }
}

FileStream::FileStream(
    std::string const& directory,
    std::string const& base_filename,
    std::string const& extension,
    std::size_t bytes_per_file,
    HeaderUpdateCallback header_update_callback)
: _directory(directory)
, _base_filename(base_filename)
, _extension(extension)
, _bytes_per_file(bytes_per_file)
, _header_update_callback(header_update_callback)
, _total_bytes_written(0)
, _current_file(nullptr)
{
    if (_bytes_per_file == 0)
    {
        throw std::runtime_error("The number of bytes per file must be greater than zero");
    }
    BOOST_LOG_TRIVIAL(info) << "Creating output file stream with parameters,\n"
                            << "Output directory: " << _directory << "\n"
                            << "Base filename: " << _base_filename << "\n"
                            << "Extension: " << _extension << "\n"
                            << "Number of bytes per file: " << _bytes_per_file;
}

FileStream::~FileStream()
{
    if (_current_file)
    {
        _current_file.reset(nullptr);
    }
}

void FileStream::write(char const* ptr, std::size_t bytes)
{
    BOOST_LOG_TRIVIAL(debug) << "Writing " << bytes << " bytes to file stream";
    if (_current_file)
    {
        std::size_t bytes_written = _current_file->write(ptr, bytes);
        _total_bytes_written += bytes_written;
        if (bytes_written < bytes)
        {
            new_file();
            write(ptr + bytes_written, bytes - bytes_written);
        }
    }
    else
    {
        new_file();
        write(ptr, bytes);
    }
}

void FileStream::new_file()
{
    std::stringstream full_path;
    full_path << _directory << "/" << _base_filename
              << "_" << std::setfill('0') << std::setw(16) << _total_bytes_written
              << std::setfill(' ') << _extension;
    std::size_t header_bytes;
    BOOST_LOG_TRIVIAL(debug) << "Retrieving updated header";
    // The callback needs to guarantee the lifetime of the returned pointer here
    std::shared_ptr<char const> header_ptr = _header_update_callback(
        header_bytes, _total_bytes_written);
    _current_file.reset(new File(full_path.str(), _bytes_per_file + header_bytes));
    //Here we are directly invoking the write method on the File object
    //to avoid potential bugs when the header is not completely written
    BOOST_LOG_TRIVIAL(debug) << "Writing updated header";
    if (_current_file->write(header_ptr.get(), header_bytes) != header_bytes)
    {
        throw std::runtime_error("Unable to write header to File instance");
    }
}

} // psrdada_cpp
