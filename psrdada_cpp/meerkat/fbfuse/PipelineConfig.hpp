#ifndef PSRDADA_CPP_MEERKAT_PIPELINECONFIG_HPP
#define PSRDADA_CPP_MEERKAT_PIPELINECONFIG_HPP

class PipelineConfig
{
public:
    PipelineConfig();
    ~PipelineConfig();
    PipelineConfig(PipelineConfig const&) = delete;

    std::string const& delay_buffer_shm() const;
    void delay_buffer_shm(std::string const&);

    std::string const& delay_buffer_mutex() const;
    void delay_buffer_mutex(std::string const&);

    std::string const& delay_buffer_sem() const;
    void delay_buffer_sem(std::string const&);

public:

};

#endif //PSRDADA_CPP_MEERKAT_PIPELINECONFIG_HPP