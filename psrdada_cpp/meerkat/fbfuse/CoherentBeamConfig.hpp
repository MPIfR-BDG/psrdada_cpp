#ifndef PSRDADA_CPP_MEERKAT_COHERENTBEAMCONFIG_HPP
#define PSRDADA_CPP_MEERKAT_COHERENTBEAMCONFIG_HPP

#include <vector>
#include <map>
#include <pair>
#include <string>

class CoherentBeamConfig
{
public:
    typedef std::map<std::string, Beam> BeamMap;

private:
    BeamMap _beam_map;
    std::vector<std::string> _antenna_ids;
    std::size_t _taccumulate;
    std::size_t _faccumulate;
    bool enabled;

public:
    CoherentBeamConfig();
    CoherentBeamConfig(std::vector<Beam> const& beams);
    ~CoherentBeamConfig();

    void add_beam(std::string const& id, RaDecCoordinates const& coordinates);
    BeamMap const& beam_map() const;

    std::vector<std::string> const& antenna_ids() const;
    void antenna_ids(std::vector<std::string> const& ants);

    void taccumulate(std::size_t tacc);
    std::size_t taccumulate() const;

    void faccumulate(std::size_t facc);
    std::size_t faccumulate() const;
};

#endif //PSRDADA_CPP_MEERKAT_COHERENTBEAMCONFIG_HPP