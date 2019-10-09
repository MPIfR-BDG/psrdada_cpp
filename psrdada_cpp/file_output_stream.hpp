#ifndef PSRDADA_CPP_FILE_OUTPUT_STREAM_HPP
#define PSRDADA_CPP_FILE_OUTPUT_STREAM_HPP

#include "psrdada_cpp/common.hpp"
#include <fstream>
#include <functional>

namespace psrdada_cpp {

class FileStream
{
public:
    typedef std::function<std::shared_ptr<char const>(std::size_t&, std::size_t)> HeaderUpdateCallback;

private:

    class File
    {
    public:
        /**
         * @brief      Internal class for managing size capped output streams
         *
         * @param      fname  The filename to write to (full path)
         * @param[in]  bytes  The maximum number of bytes that can be written to the file.
         */
        File(std::string const& fname, std::size_t bytes);
        File(File const&) = delete;
        ~File();

        /**
         * @brief      Write binary data to the file
         *
         * @param      ptr    Pointer to the memory to write
         * @param[in]  bytes  The number of bytes to write
         *
         * @note       If the maximume number of bytes has
         *
         * @return     The number of bytes written in this execution
         */
        std::size_t write(char const* ptr, std::size_t bytes);

    private:
        std::string _full_path;
	std::size_t _bytes_requested;
        std::size_t _bytes_written;
        std::ofstream _stream;
    };
public:

    /**
     * @brief      An object that manages writing a stream of data to
     *             multiple formatted, size-capped files. The formatting of
     *             the files written is always [header][data].
     *
     * @param      directory       The directory to write to
     * @param      base_filename   The base filename of the files to write
     * @param      extension       The file extension to use
     * @param[in]  header_updater  A callback used to get updates to the header so that
     *                             it can be kept up to date based on the amount of data
     *                             written.
     */
    explicit FileStream(
        std::string const& directory,
        std::string const& base_filename,
        std::string const& extension,
        std::size_t bytes_per_file,
        HeaderUpdateCallback header_updater);
    FileStream(FileStream const&) = delete;
    ~FileStream();

    /**
     * @brief      Write a block of data to the stream
     *
     * @param      ptr    A pointer to the memory to write
     * @param[in]  bytes  The number of bytes to write
     *
     * @note  If this write corresponds to the initial write, or the
     *        first write after a new file has been generated, the header
     *        update callback will be executed to obtain a valid header which
     *        will be written to the file before any data.
     */
    void write(char const* ptr, std::size_t bytes);

private:
    void new_file();

private:
    std::string const _directory;
    std::string const _base_filename;
    std::string const _extension;
    std::size_t _bytes_per_file;
    HeaderUpdateCallback _header_update_callback;
    std::size_t _total_bytes_written;
    std::unique_ptr<File> _current_file;
};

}

#endif //PSRDADA_CPP_FILE_OUTPUT_STREAM_HPP
