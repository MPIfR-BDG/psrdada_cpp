#ifndef PSRDADA_CPP_MEERKAT_FBFUSE_DELAYENGINESIMULATOR_HPP
#define PSRDADA_CPP_MEERKAT_FBFUSE_DELAYENGINESIMULATOR_HPP

#include "psrdada_cpp/meerkat/fbfuse/PipelineConfig.hpp"
#include "psrdada_cpp/meerkat/fbfuse/DelayModel.cuh"
#include <semaphore.h>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {

class DelayEngineSimulator
{
public:
    /**
     * @brief      Create a new DelayEngineSimulator object
     *
     * @param      config  The pipeline configuration.
     *
     */
    explicit DelayEngineSimulator(PipelineConfig const& config);
    ~DelayEngineSimulator();
    DelayEngineSimulator(DelayEngineSimulator const&) = delete;

    /**
     * @brief      Simulate an update to the delay model by the control system
     */
    void update_delays();

    /**
     * @brief      Return a pointer to the delay model
     */
    DelayModel* delay_model();

private:
    std::string const _delay_buffer_shm;
    std::string const _delay_buffer_sem;
    std::string const _delay_buffer_mutex;
    int _shm_fd;
    void* _shm_ptr;
    sem_t* _sem_id;
    sem_t* _mutex_id;
    DelayModel* _delay_model;
};

} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

#endif // PSRDADA_CPP_MEERKAT_FBFUSE_DELAYENGINESIMULATOR_HPP



