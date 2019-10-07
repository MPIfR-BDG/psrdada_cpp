#ifndef PSRDADA_CPP_SIGPROC_FILE_WRITER_HPP
#define PSRDADA_CPP_SIGPROC_FILE_WRITER_HPP

#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/sigprocheader.hpp"
#include "psrdada_cpp/file_output_stream.hpp"
#include <memory>
#include <string>

namespace psrdada_cpp {

/*
 * @brief: A file writer for outputing SigProc format files.
 */

class SigprocFileWriter
{
private:
    /**
     * @brief      Indicates whether files will be output to disk or not.
     */
    enum State
    {
        ENABLED,
        DISABLED
    };

public:

    /**
     * @brief      Create a new file writer for sigproc-like data.
     */
    SigprocFileWriter();
    SigprocFileWriter(SigprocFileWriter const&) = delete;
    ~SigprocFileWriter();

    /**
     * @brief      Set a tag to be used in the filename of generated files
     *
     * @detail     The tag will be inserted into the file name as '_<tag>_'.
     *             An example use would be to include the beam id of beam being
     *             written.
     *
     * @param      tag_  The tag
     */
    void tag(std::string const& tag_);

    /**
     * @brief      Get the file tag
     *
     * @return     The tag
     */
    std::string const& tag() const;

    /**
     * @brief      Set the output directory
     *
     * @param      dir   The directory
     */
    void directory(std::string const& dir);

    /**
     * @brief      Get the output directory
     *
     * @return     The directory
     */
    std::string const& directory() const;

    /**
     * @brief      Set the maximum file size for generated file.
     *
     * @detail     This value does not include the size of the headers
     *             that will be written to the file (so payload only).
     *
     * @param[in]  size  The size in bytes
     */
    void max_filesize(std::size_t size);

    /**
     * @brief      Get the maximum file size.
     *
     * @return     The size in bytes.
     */
    std::size_t max_filesize() const;

    /**
     * @brief      Init function for use in handler chains
     *
     * @param      block  A RawBytes object containing metadata in sigproc format
     */
    void init(RawBytes& block);

    /**
     * @brief      Call function for use in handler chains
     *
     * @param      block  A RawBytes object containing data in sigproc format
     */
    bool operator()(RawBytes&);

    /**
     * @brief      Enable the writer
     *
     * @detail     Calling this will change the writer state to enabled, and if necessary
     *             set flags to indicate that a new file stream is required. Changes
     *             will only come into effect on the next call to the operator() method.
     *
     * @note       Has no effect if the writer is already enabled. To trigger a new
     *             stream the writer must first be disabled then enabled.
     */
    void enable();

    /**
     * @brief      Disable the writer.
     *
     * @detail     Calling this will change the writer state to disabled. If there is an
     *             active file stream the stream will be flushed to disk and closed. Changes
     *             will only come into effect on the next call to the operator() method. If
     *             and enable call comes before the next call to the operator method the system
     *             will simply generate a new stream when the operator() method is called (unless
     *             another disable comes in before that... and so on).
     *
     * @note       Has no effect on an already disabled writer.
     */
    void disable();

    /**
     * @brief      Get the current filterbank header information
     *
     * @detail     Intended to allow external access to the header for the
     *             purpose of on-the-fly modification.
     *
     * @return     A reference to the FilHead object storing the observation metadata
     */
    FilHead& header();

private:
    void new_stream();

private:
    std::size_t _max_filesize;
    State _state;
    std::string _tag;
    std::string _outdir;
    std::string _extension;
    std::size_t _total_bytes;
    bool _new_stream_required;
    FilHead _header;
    std::unique_ptr<FileStream> _stream;
};
}//psrdada_cpp

#endif //PSRDADA_CPP_SIGPROC_FILE_WRITER_HPP
