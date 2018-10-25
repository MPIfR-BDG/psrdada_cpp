#include "psrdada_cpp/meerkat/fbfuse/DelayManager.cuh"
#include "psrdada_cpp/cuda_utils.hpp"
#include <sys/mman.h>
#include <sys/stat.h>        /* For mode constants */
#include <fcntl.h>           /* For O_* constants */
#include <errno.h>
#include <cstring>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {

DelayManager::DelayManager(PipelineConfig const& config, cudaStream_t stream)
    : _config(config)
    , _copy_stream(stream)
    , _last_sem_value(0)
{
    BOOST_LOG_TRIVIAL(debug) << "Constructing new DelayManager instance";
    BOOST_LOG_TRIVIAL(debug) << "Opening delay buffer shared memory segement";
    // First we open file descriptors to all shared memory segments and semaphores
    _delay_buffer_fd = shm_open(_config.delay_buffer_shm().c_str(), O_RDONLY, 0);
    if (_delay_buffer_fd == -1)
    {
        throw std::runtime_error(std::string(
            "Failed to open delay buffer shared memory: "
            ) + std::strerror(errno));
    }
    BOOST_LOG_TRIVIAL(debug) << "Opening delay buffer mutex semaphore";
    _delay_mutex_sem = sem_open(_config.delay_buffer_mutex().c_str(), O_EXCL);
    if (_delay_mutex_sem == SEM_FAILED)
    {
        throw std::runtime_error(std::string(
            "Failed to open delay buffer mutex semaphore: "
            ) + std::strerror(errno));
    }
    BOOST_LOG_TRIVIAL(debug) << "Opening delay buffer counting semaphore";
    _delay_count_sem = sem_open(_config.delay_buffer_sem().c_str(), O_EXCL);
    if (_delay_count_sem == SEM_FAILED)
    {
        throw std::runtime_error(std::string(
            "Failed to open delay buffer counting semaphore: "
            ) + std::strerror(errno));
    }

    // Here we run fstat on the shared memory buffer to check that it is the right dimensions
    BOOST_LOG_TRIVIAL(debug) << "Verifying shared memory segment dimensions";
    struct stat mem_info;
    int retval = fstat(_delay_buffer_fd, &mem_info);
    if (retval == -1)
    {
        throw std::runtime_error(std::string(
            "Could not fstat the delay buffer shared memory: ")
            + std::strerror(errno));
    }
    if (mem_info.st_size != sizeof(DelayModel))
    {
        throw std::runtime_error(std::string(
            "Shared memory buffer had unexpected size: ")
            + std::to_string(mem_info.st_size));
    }

    // Here we memory map the buffer and cast to the expected format (DelayModel POD struct)
    BOOST_LOG_TRIVIAL(debug) << "Memory mapping shared memory segment";
    _delay_model = static_cast<DelayModel*>(mmap(NULL, sizeof(DelayModel), PROT_READ,
        MAP_SHARED, _delay_buffer_fd, 0));
    if (_delay_model == NULL)
    {
        throw std::runtime_error(std::string(
            "MMAP on delay model buffer returned a null pointer: ")
            + std::strerror(errno));
    }

    // To maximise the copy throughput for the delays we here register the host memory
    BOOST_LOG_TRIVIAL(debug) << "Registering shared memory segement with CUDA driver";
    CUDA_ERROR_CHECK(cudaHostRegister(static_cast<void*>(_delay_model->delays),
        sizeof(_delay_model->delays), cudaHostRegisterMapped));

    // Resize the GPU array for the delays
    _delays.resize(FBFUSE_CB_NBEAMS * FBFUSE_CB_NANTENNAS);
}

DelayManager::~DelayManager()
{
    BOOST_LOG_TRIVIAL(debug) << "Destroying DelayManager instance";
    CUDA_ERROR_CHECK(cudaHostUnregister(static_cast<void*>(_delay_model->delays)));
    munmap(_delay_model, sizeof(DelayModel));
    close(_delay_buffer_fd);
    sem_close(_delay_mutex_sem);
    sem_close(_delay_count_sem);
}

bool DelayManager::update_available()
{
    BOOST_LOG_TRIVIAL(debug) << "Checking for delay model update";
    int count;
    int retval = sem_getvalue(_delay_count_sem, &count);
    if (retval != 0)
    {
        throw std::runtime_error(std::string(
            "Unable to retrieve value of counting semaphore: ")
            + std::strerror(errno));
    }
    if (count == _last_sem_value)
    {
        BOOST_LOG_TRIVIAL(debug) << "No delay model update available";
        return false;
    }
    else
    {
        BOOST_LOG_TRIVIAL(debug) << "New delay model avialable";
        if (_last_sem_value - count > 1)
        {
            // This implies that there has been an update since the function was last called and
            // we need to trigger a memcpy between the host and the device. This should acquire
            // the mutex during the copy.
            // We also check if we have somehow skipped and update.
            BOOST_LOG_TRIVIAL(warning) << "Semaphore value increased by " << (_last_sem_value - count)
            << " between checks (exepcted increase of 1)";
        }
        _last_sem_value = count;
        return true;
    }
}

DelayManager::DelayVectorType const& DelayManager::delays()
{
    // This function should return the delays in GPU memory
    // First check if we need to update GPU memory
    if (update_available())
    {
        // Block on mutex semaphore
        // Technically this should *never* block as an increment to the
        // counting semaphore implies that the delay model has been updated
        // already. This is merely here for safety but may be removed in future.
        BOOST_LOG_TRIVIAL(debug) << "Acquiring shared memory mutex";
        int retval = sem_wait(_delay_mutex_sem);
        if (retval != 0)
        {
            throw std::runtime_error(std::string(
                "Unable to wait on mutex semaphore: ")
            + std::strerror(errno));
        }
        // Although this is intended as a blocking copy, it should only block on the host, not the GPU,
        // as such we use an async memcpy in a dedicated stream.
        void* dst = static_cast<void*>(thrust::raw_pointer_cast(_delays.data()));
        BOOST_LOG_TRIVIAL(debug) << "Copying delays to GPU";
        CUDA_ERROR_CHECK(cudaMemcpyAsync(dst, (void*) _delay_model->delays, sizeof(_delay_model->delays),
            cudaMemcpyHostToDevice, _copy_stream));
        CUDA_ERROR_CHECK(cudaStreamSynchronize(_copy_stream));

        BOOST_LOG_TRIVIAL(debug) << "Releasing shared memory mutex";
        retval = sem_post(_delay_mutex_sem);
        if (retval != 0)
        {
            throw std::runtime_error(std::string(
                "Unable to release mutex semaphore: ")
            + std::strerror(errno));
        }
    }
    return _delays;
}

} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp
