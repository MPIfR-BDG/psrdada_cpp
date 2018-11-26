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

    void update_delays();

private:
    PipelineConfig const& _config;
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



