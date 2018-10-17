Remember include guards

#include "psrdada_cpp/meerkat/fbfuse/PipelineConfig.hpp"
#include "psrdada_cpp/meerkat/fbfuse/DelayClient.hpp"
#include <thrust/device_vector.h>
#include <sys/mman.h>
#include <sys/stat.h>        /* For mode constants */
#include <fcntl.h>           /* For O_* constants */
#include <errno.h>
#include <semaphore.h>


// POD struct containing the layout of the shared memory
// buffer as written by the Python client
struct DelayModel
{
    double epoch;
    double duration;
    float2 delays[NBEAMS * NANTENNAS];
};

class DelayManager
{
public:
    typedef float2 DelayType;
    typedef double TimeType;

public:
    DelayManager(PipelineConfig const& config);
    ~DelayManager();
    DelayManager(DelayManager const&) == delete;

    //returns the delays as GPU memory
    DelayType const* delays(TimeType epoch);

private:
    int get_count() const;

private:
    PipelineConfig const& _config;
    cudaStream_t _copy_stream;
    int _delay_buffer_fd;
    sem_t _delay_mutex_sem;
    sem_t _delay_count_sem;
    int _last_sem_value;
    DelayModel* _delay_model;
    thrust::device_vector<WeightsType> _delays;
};


DelayManager::DelayManager(PipelineConfig const& config)
    : _config(config)
    , _delay_model(NULL)
{
    // First we open file descriptors to all shared memory segments and semaphores
    _delay_buffer_fd = shm_open(_config.delay_buffer_shm().c_str(), O_RDONLY, 0);
    _delay_mutex_sem = sem_open(_config.delay_buffer_mutex().c_str(), O_EXCL);
    _delay_count_sem = sem_open(_config.delay_buffer_sem().c_str(), O_EXCL);

    // Here we run fstat on the shared memory buffer to check that it is the right dimensions
    struct stat mem_info;
    int retval = fstat(_delay_buffer_fd, &mem_info);
    if (retval != 0)
    {
        throw std::runtime_error(std::string("Could not fstat the delay buffer shared memory, error code: ") + std::to_string(retval));
    }

    if (mem_info.st_size != sizeof(DelayModel))
    {
        throw std::runtime_error(std::string("Shared memory buffer had unexpected size: ") + std::to_string(mem_info.st_size));
    }

    // Here we memory map the buffer and cast to the expected format
    _delay_model = static_cast<DelayManager*>(mmap(NULL, sizeof(DelayModel), PROT_READ, MAP_SHARED, _delay_buffer_fd, 0));
    if (_delay_model == NULL)
    {
        throw std::runtime_error(std::string("MMAP on delay model buffer returned a null pointer with errno: ") + std::to_string(errno));
    }

    // To maximise the copy throughput for the delays we here register the host memory
    CUDA_SAFE_CALL(cudaHostRegister(static_cast<void*>(_delay_model.delays), sizeof(_delay_model.delays), cudaHostRegisterMapped));

    // Resize the GPU array for the delays
    _delays.resize(NBEAMS * NANTENNAS);
}

DelayManager::~DelayManager()
{
    munmap(_delay_model, sizeof(DelayModel));
    close(_delay_buffer_fd);
    sem_close(_delay_mutex_sem);
    sem_close(_delay_count_sem);
}

int DelayManager::get_count()
{
    int count;
    int retval = sem_getvalue(_delay_count_sem, &count);
    if (retval != 0)
    {
        throw std::runtime_error(std::string("Unable to retrieve value of counting semaphore, with errno: ") + std::string(errno));
    }
    return count;
}

DelayManager::DelayType const* delays()
{
    // This function should return the delays in GPU memory
    // First check if we need to update GPU memory
    int sem_value = get_count();
    if (sem_value != _last_sem_value)
    {
        // This implies that there has been an update since the function was last called and
        // we need to trigger a memcpy between the host and the device. This should acquire
        // the mutex during the copy.
        // We also check if we have somehow skipped and update.
        if (_last_sem_value - semaphore > 1)
        {
            BOOST_LOG_WARN << "Semaphore value increased by " << (_last_sem_value - semaphore) << " between calls to delays()";
        }

        // Block on mutex semaphore
        // Technically this should *never* block as an increment to the
        // counting semaphore implies that the delay model has been updated
        // already. This is merely here for safety but may be removed in future.
        int retval = sem_wait(_delay_mutex_sem);
        if (retval != 0)
        {
            throw std::runtime_error(std::string("Unable to wait on mutex semaphore, with errno: ") + std::string(errno));
        }

        // Although this is intended as a blocking copy, it should only block on the host, not the GPU,
        // as such we use an async memcpy in a dedicated stream.
        void* dst = static_cast<void*>(thrust::raw_pointer_cast(_delays.data()));
        CUDA_SAFE_CALL(cudaMemcpyAsync(dst, (void*) _delay_model.delays, sizeof(_delay_model.delays), cudaMemcpyHostToDevice, _copy_stream));
        CUDA_SAFE_CALL(cudaStreamSynchronize(_copy_stream));

        int retval = sem_post(_delay_mutex_sem);
        if (retval != 0)
        {
            throw std::runtime_error(std::string("Unable to release mutex semaphore, with errno: ") + std::string(errno));
        }
    }
    return thrust::raw_pointer_cast(_delays.data());

}


