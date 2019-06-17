#include "psrdada_cpp/simple_shm_writer.hpp"
#include <cassert>

namespace psrdada_cpp {

SimpleShmWriter::SimpleShmWriter(
    std::string const& shm_key,
    std::size_t header_size,
    std::size_t data_size)
    : _shm_key(shm_key)
    , _header_size(header_size)
    , _data_size(data_size)
{
    _shm_fd = shm_open(_shm_key.c_str(), O_CREAT | O_EXCL | O_RDWR, 0666);
    if (_shm_fd == -1)
    {
        std::stringstream msg;
        msg << "Failed to open shared memory named "
        << _shm_key << " with error: "
        << std::strerror(errno);
        throw std::runtime_error(msg.str());
    }
    if (ftruncate(_shm_fd, _header_size + _data_size) == -1)
    {
        std::stringstream msg;
        msg << "Failed to ftruncate shared memory named "
        << _shm_key << " with error: "
        << std::strerror(errno);
        throw std::runtime_error(msg.str());
    }
    _shm_ptr = mmap(0, _header_size + _data_size, PROT_WRITE, MAP_SHARED, _shm_fd, 0);
    if (_shm_ptr == NULL)
    {
        std::stringstream msg;
        msg << "Failed to mmap shared memory named "
        << _shm_key << " with error: "
        << std::strerror(errno);
        throw std::runtime_error(msg.str());
    }
}

SimpleShmWriter::~SimpleShmWriter()
{
    if (munmap(_shm_ptr, _header_size + _data_size) == -1)
    {
        std::stringstream msg;
        msg << "Failed to unmap shared memory "
        << _shm_key << " with error: "
        << std::strerror(errno);
        throw std::runtime_error(msg.str());
    }

    if (close(_shm_fd) == -1)
    {
        std::stringstream msg;
        msg << "Failed to close shared memory file descriptor "
        << _shm_fd << " with error: "
        << std::strerror(errno);
        throw std::runtime_error(msg.str());
    }

    if (shm_unlink(_shm_key.c_str()) == -1)
    {
        std::stringstream msg;
        msg << "Failed to unlink shared memory "
        << _shm_key << " with error: "
        << std::strerror(errno);
        throw std::runtime_error(msg.str());
    }
}

void SimpleShmWriter::init(RawBytes& block)
{
    assert(block.used_bytes() == _header_size);
    std::memcpy(_shm_ptr, static_cast<void*>(block.ptr()), _header_size);
}

bool SimpleShmWriter::operator()(RawBytes& block)
{
    assert(block.used_bytes() == _data_size);
    std::memcpy(static_cast<void*>(static_cast<char*>(_shm_ptr) + _header_size),
        static_cast<void*>(block.ptr()), _data_size);
    return false;
}

} //psrdada_cpp
