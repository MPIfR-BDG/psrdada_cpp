#ifndef PSRDADA_CPP_MEERKAT_DELAYMODEL_HPP
#define PSRDADA_CPP_MEERKAT_DELAYMODEL_HPP

//This only includes delays for antennas to be added into a coherent beam sum.

struct poly_t
{
    double coeff1;
    double coeff0;
};

/**
 * @brief [brief description]
 * @details
 *
 */
class DelayModel
{
private:
    friend class DelayEngineClient;

private:
    // order here should be [beam, antenna];
    std::map<std::string, unsigned> beam_map;
    std::map<std::string, unsigned> antenna_map;
    time_t _tstart;
    time_t _tfinish;
    std::vector<poly_t> _coeffs;
    std::unique_ptr<std::mutex> _mutex;
    CoherentBeamConfig const& _config;

    std::mutex& mutex() const;

public:
    DelayModel(CoherentBeamConfig const& config);
    ~DelayModel();

    void validity(time_t tstart, time_t tfinish);
    std::pair<time_t, time_t> validity() const;

    std::size_t nantennas() const;

    std::size_t nbeams() const;

    //requires a mutex lock
    std::vector<poly_t> const& coefficients() const;

    void update(std::string beam_id, std::string antenna_id,
        double delay_rate, double delay_offset);
};

#endif //PSRDADA_CPP_MEERKAT_DELAYMODEL_HPP