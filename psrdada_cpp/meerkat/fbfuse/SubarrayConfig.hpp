#ifndef PSRDADA_CPP_MEERKAT_SUBARRAYCONFIG_HPP
#define PSRDADA_CPP_MEERKAT_SUBARRAYCONFIG_HPP

#include <vector>
#include <map>
#include <string>

class SubarrayConfig
{
public:
    typedef std::map<std::string, Antenna> AntennaMap;

private:
    AntennaMap _antenna_map;
    EcefCoordinates _phase_ref;
    frequency_t _centre_frequency;
    frequency_t _bandwidth;
    std::vector<frequency_t> _channel_frequencies;
    unsigned _nchannels;

public:
    SubarrayConfig();
    SubarrayConfig(std::vector<Antenna> const& ants, EcefCoordinates const& phref,
        frequency_t cfreq, frequency_t bw, unsigned nchans);
    ~SubarrayConfig();

    AntennaMap const& antenna_map() const;
    void add_antenna(std::string const& name, EcefCoordinates const& coordinates);

    EcefCoordinates const& phase_reference() const;
    void phase_reference(EcefCoordinates const& phref);

    frequency_t centre_frequency() const;
    void centre_frequency(frequency_t cfreq);

    frequency_t bandwidth() const;
    void bandwidth(frequency_t cfreq);

    unsigned nchannels() const;
    void nchannels(unsigned nchans);

    std::vector<frequency_t> const& channel_frequencies() const;

    Antenna const& antenna(std::string);
};

#endif //PSRDADA_CPP_MEERKAT_SUBARRAYCONFIG_HPP