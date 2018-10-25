#ifndef PSRDADA_CPP_MEERKAT_FBFUSE_DELAYMANAGER_HPP
#define PSRDADA_CPP_MEERKAT_FBFUSE_DELAYMANAGER_HPP

#include "psrdada_cpp/meerkat/fbfuse/PipelineConfig.hpp"
#include <thrust/device_vector.h>
#include <semaphore.h>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {

// POD struct containing the layout of the shared memory
// buffer as written by the Python client
struct DelayModel
{
    double epoch;
    double duration;
    float2 delays[FBFUSE_CB_NBEAMS * FBFUSE_CB_NANTENNAS]; // Compile time constants
};

/**
 * @brief      Class for managing the POSIX shared memory buffers
 *             and semaphores wrapping the delay model updates from
 *             the control system.
 */
class DelayManager
{
public:
    typedef float2 DelayType;
    typedef thrust::device_vector<DelayType> DelayVectorType;
    typedef double TimeType;

public:
    /**
     * @brief      Create a new DelayManager object
     *
     * @param      config  The pipeline configuration.
     *
     * @detail     The passed pipeline configuration contains the names
     *             of the POSIX shm and sem to connect to for the delay
     *             models.
     */
    DelayManager(PipelineConfig const& config, cudaStream_t stream);
    ~DelayManager();
    DelayManager(DelayManager const&) = delete;

    /**
     * @brief      Get the current delay model
     *
     * @detail     On a call to this function, a check is made on the
     *             delays counting semaphore to see if a delay model
     *             update is available. If so, the values are retrieved
     *             from shared memory and copied to the GPU. This function
     *             is not thread-safe!!!
     *
     * @return     A device vector containing the current delays
     */
    DelayVectorType const& delays();

    //needs implemented
    //double epoch() const;
    //double duration() const;

private:
    bool update_available();

private:
    PipelineConfig const& _config;
    DelayModel* _delay_model;
    cudaStream_t _copy_stream;
    int _delay_buffer_fd;
    sem_t* _delay_mutex_sem;
    sem_t* _delay_count_sem;
    int _last_sem_value;
    DelayVectorType _delays;
};

} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

#endif // PSRDADA_CPP_MEERKAT_FBFUSE_DELAYMANAGER_HPP



