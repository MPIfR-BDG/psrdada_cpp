#include "psrdada_cpp/meerkat/fbfuse/DelayEngineSimulator.hpp"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <cstring>
#include <sys/mman.h>
#include <sstream>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {

DelayEngineSimulator::DelayEngineSimulator(PipelineConfig const& config)
    : _config(config)
{
    _shm_fd = shm_open(_config.delay_buffer_shm().c_str(), O_CREAT | O_RDWR, 0666);
    if (_shm_fd == -1)
    {
        std::stringstream msg;
        msg << "Failed to open shared memory named "
        << _config.delay_buffer_shm() << " with error: "
        << std::strerror(errno);
        throw std::runtime_error(msg.str());
    }
    if (ftruncate(_shm_fd, sizeof(DelayModel)) == -1)
    {
        std::stringstream msg;
        msg << "Failed to ftruncate shared memory named "
        << _config.delay_buffer_shm() << " with error: "
        << std::strerror(errno);
        throw std::runtime_error(msg.str());
    }
    _shm_ptr = mmap(0, sizeof(DelayModel), PROT_WRITE, MAP_SHARED, _shm_fd, 0);
    if (_shm_ptr == NULL)
    {
        std::stringstream msg;
        msg << "Failed to mmap shared memory named "
        << _config.delay_buffer_shm() << " with error: "
        << std::strerror(errno);
        throw std::runtime_error(msg.str());
    }
    _delay_model = static_cast<DelayModel*>(_shm_ptr);
    _sem_id = sem_open(_config.delay_buffer_sem().c_str(), O_CREAT, 0666, 0);
    if (_sem_id == SEM_FAILED)
    {
        std::stringstream msg;
        msg << "Failed to open delay buffer semaphore "
        << _config.delay_buffer_sem() << " with error: "
        << std::strerror(errno);
        throw std::runtime_error(msg.str());
    }
    _mutex_id = sem_open(_config.delay_buffer_mutex().c_str(), O_CREAT, 0666, 0);
    if (_mutex_id == SEM_FAILED)
    {
        std::stringstream msg;
        msg << "Failed to open delay buffer mutex "
        << _config.delay_buffer_mutex() << " with error: "
        << std::strerror(errno);
        throw std::runtime_error(msg.str());
    }
    // Here we post once so that the mutex has a value of 1
    // and can so be safely acquired
    sem_post(_mutex_id);
}

DelayEngineSimulator::~DelayEngineSimulator()
{
    if (munmap(_shm_ptr, sizeof(DelayModel)) == -1)
    {
        std::stringstream msg;
        msg << "Failed to unmap shared memory "
        << _config.delay_buffer_shm() << " with error: "
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

    if (shm_unlink(_config.delay_buffer_shm().c_str()) == -1)
    {
        std::stringstream msg;
        msg << "Failed to unlink shared memory "
        << _config.delay_buffer_shm() << " with error: "
        << std::strerror(errno);
        throw std::runtime_error(msg.str());
    }

    if (sem_close(_sem_id) == -1)
    {
        std::stringstream msg;
        msg << "Failed to close semaphore "
        << _config.delay_buffer_sem() << " with error: "
        << std::strerror(errno);
        throw std::runtime_error(msg.str());
    }

    if (sem_close(_mutex_id) == -1)
    {
        std::stringstream msg;
        msg << "Failed to close mutex "
        << _config.delay_buffer_mutex() << " with error: "
        << std::strerror(errno);
        throw std::runtime_error(msg.str());
    }
}

void DelayEngineSimulator::update_delays()
{
    sem_post(_sem_id);
}

} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp