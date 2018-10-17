#include "psrdada_cpp/meerkat/fbfuse/PipelineConfig.hpp"
#include "psrdada_cpp/meerkat/fbfuse/DelayClient.hpp"
#include <thrust/device_vector.h>

class WeightsManager
{
public:
    typedef char2 WeightsType;
    typedef double TimeType;

public:
    WeightsManager(PipelineConfig const& config);
    ~WeightsManager();
    WeightsManager(WeightsManager const&) == delete;

    WeightsType const* weights(TimeType epoch);

private:
    std::unique_ptr<DelayClient> _delay_client;
    TimeType _update_rate;
    TimeType _last_update;
    thrust::device_vector<WeightsType> _weights;
};


