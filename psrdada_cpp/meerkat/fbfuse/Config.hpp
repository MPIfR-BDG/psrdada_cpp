#ifndef PSRDADA_CPP_MEERKAT_CONFIG_HPP
#define PSRDADA_CPP_MEERKAT_CONFIG_HPP

class Config
{
private:
    SubarrayConfig _subarray_config;
    CoherentBeamConfig _coherent_beam_config;
    IncoherentBeamConfig _incoherent_beam_config;
    PipelineConfig _pipeline_config;

public:
    Config();
    ~Config();

    SubarrayConfig& subarray_config();
    CoherentBeamConfig& coherent_beam_config();
    IncoherentBeamConfig& incoherent_beam_config();
    PipelineConfig& pipeline_config();
};

#endif //PSRDADA_CPP_MEERKAT_CONFIG_HPP