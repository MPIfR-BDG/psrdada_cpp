#include "psrdada_cpp/meerkat/fbfuse/test/DelayManagerTester.cuh"
#include "psrdada_cpp/meerkat/fbfuse/fbfuse_constants.hpp"
#include "psrdada_cpp/cuda_utils.hpp"
#include "thrust/host_vector.hpp"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <cstring>
#include <sys/mman.h>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace test {

DelayManagerTester::DelayManagerTester()
    : ::testing::Test()
    , _shm_fd(0)
    , _shm_ptr(nullptr)
    , _sem_id(nullptr)
    , _mutex_id(nullptr)
    , _delay_model(nullptr)
    , _stream(0)
{
    _config.delay_buffer_shm("test_delay_buffer_shm");
    _config.delay_buffer_sem("test_delay_buffer_sem");
    _config.delay_buffer_mutex("test_delay_buffer_mutex");
}

DelayManagerTester::~DelayManagerTester()
{

}

void DelayManagerTester::SetUp()
{
    _shm_fd = shm_open(_config.delay_buffer_shm().c_str(), O_CREAT | O_RDWR, 0666);
    if (_shm_fd == -1)
    {
        FAIL() << "Failed to open shared memory named "
        << _config.delay_buffer_shm() << " with error: "
        << std::strerror(errno);
    }
    if (ftruncate(_shm_fd, sizeof(DelayModel)) == -1)
    {
        FAIL() << "Failed to ftruncate shared memory named "
        << _config.delay_buffer_shm() << " with error: "
        << std::strerror(errno);
    }
    _shm_ptr = mmap(0, sizeof(DelayModel), PROT_WRITE, MAP_SHARED, _shm_fd, 0);
    if (_shm_ptr == NULL)
    {
        FAIL() << "Failed to mmap shared memory named "
        << _config.delay_buffer_shm() << " with error: "
        << std::strerror(errno);
    }
    _delay_model = static_cast<DelayModel*>(_shm_ptr);
    _sem_id = sem_open(_config.delay_buffer_sem().c_str(), O_CREAT, 0666, 0);
    if (_sem_id == SEM_FAILED)
    {
        FAIL() << "Failed to open delay buffer semaphore "
        << _config.delay_buffer_sem() << " with error: "
        << std::strerror(errno);
    }
    _mutex_id = sem_open(_config.delay_buffer_mutex().c_str(), O_CREAT, 0666, 0);
    if (_mutex_id == SEM_FAILED)
    {
        FAIL() << "Failed to open delay buffer mutex "
        << _config.delay_buffer_mutex() << " with error: "
        << std::strerror(errno);
    }
    // Here we post once so that the mutex has a value of 1
    // and can so be safely acquired by the DelayManger
    sem_post(_mutex_id);
    CUDA_ERROR_CHECK(cudaStreamCreate(&_stream));
}

void DelayManagerTester::TearDown()
{
    if (munmap(_shm_ptr, sizeof(DelayModel)) == -1)
    {
        FAIL() << "Failed to unmap shared memory "
        << _config.delay_buffer_shm() << " with error: "
        << std::strerror(errno);
    }

    if (close(_shm_fd) == -1)
    {
        FAIL() << "Failed to close shared memory file descriptor "
        << _shm_fd << " with error: "
        << std::strerror(errno);
    }

    if (shm_unlink(_config.delay_buffer_shm().c_str()) == -1)
    {
        FAIL() << "Failed to unlink shared memory "
        << _config.delay_buffer_shm() << " with error: "
        << std::strerror(errno);
    }

    if (sem_close(_sem_id) == -1)
    {
        FAIL() << "Failed to close semaphore "
        << _config.delay_buffer_sem() << " with error: "
        << std::strerror(errno);
    }

    if (sem_close(_mutex_id) == -1)
    {
        FAIL() << "Failed to close mutex "
        << _config.delay_buffer_mutex() << " with error: "
        << std::strerror(errno);
    }
    CUDA_ERROR_CHECK(cudaStreamDestroy(_stream));
}

void DelayManagerTester::compare_against_host(DelayManager::DelayVectorType const& delays)
{
    // Implicit sync copy back to host
    thrust::host_vector<DelayManager::DelayType> host_delays = delays;
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    for (int ii=0; ii < FBFUSE_CB_NBEAMS * FBFUSE_CB_NANTENNAS; ++ii)
    {
        ASSERT_EQ(_delay_model->delays[ii].x, _dhost_delays[ii].x);
        ASSERT_EQ(_delay_model->delays[ii].y, _dhost_delays[ii].y);
    }
}

TEST_F(DelayManagerTester, test_updates)
{
    DelayManager delay_manager(_config, _stream);
    sem_post(_sem_id);
    auto const& delay_vector = delay_manager.delays();
    compare_against_host(delay_vector);
    std::memset(static_cast<void*>(_delay_model->delays), 1, sizeof(_delay_model->delays));
    sem_post(_sem_id);
    compare_against_host(delay_vector);
}

TEST_F(DelayManagerTester, test_bad_keys)
{
    PipelineConfig config;
    config.delay_buffer_shm("bad_test_delay_buffer_shm");
    config.delay_buffer_sem("bda_test_delay_buffer_sem");
    config.delay_buffer_mutex("bad_test_delay_buffer_mutex");
    ASSERT_THROW(DelayManager(config, _stream));
}

} //namespace test
} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

