#ifndef PSRDADA_CPP_MEERKAT_FBFUSE_DELAYMANAGER_HPP
#define PSRDADA_CPP_MEERKAT_FBFUSE_DELAYMANAGER_HPP

#include "psrdada_cpp/meerkat/fbfuse/PipelineConfig.hpp"
#include "psrdada_cpp/meerkat/fbfuse/DelayModel.cuh"
#include <thrust/device_vector.h>
#include <semaphore.h>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {

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

    /**
     * @brief      Return the epoch of the current delay model
     */
    double epoch() const;

    /**
     * @brief      Return the duration (length of validity) of the current delay model
     */
    double duration() const;

    /**
     * @brief      Request an update to the delay model
     *
     * @detail     This method will attempt to connect to a UNIX domain socket
     *             managed by the DelayBufferController instance in mpikat. This
     *             is usually found at /tmp/fbfuse_control.sock. The communication
     *             protocol over the socket is as follows:
     *             1. Connect the socket
     *             2. Send the epoch for which the delays are required as an 8-byte double
     *             3. Receive a uint8_t value indicating whether the update was successful
     *                0 = fail, 1 = success.
     *
     * @param[in]  epoch  The epoch at which the delays should be calculated
     */
    static void request_delay_model_update(std::string const& address, double epoch) const;

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



