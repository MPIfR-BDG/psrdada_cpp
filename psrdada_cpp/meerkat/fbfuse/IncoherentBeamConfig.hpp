#ifndef PSRDADA_CPP_MEERKAT_INCOHERENTBEAMCONFIG_HPP
#define PSRDADA_CPP_MEERKAT_INCOHERENTBEAMCONFIG_HPP

#include <vector>
#include <map>
#include <pair>
#include <string>

class IncoherentBeamConfig
{
private:
    std::vector<std::string> _antenna_ids;
    std::size_t _tscrunch;
    std::size_t _fscrunch;
    bool enabled;

public:
    IncoherentBeamConfig();
    ~IncoherentBeamConfig();

    std::vector<std::string> const& antenna_ids() const;
    void antenna_ids(std::vector<std::string> const& ants);

    void taccumulate(std::size_t tacc);
    std::size_t taccumulate() const;

    void faccumulate(std::size_t facc);
    std::size_t faccumulate() const;
};

#endif //PSRDADA_CPP_MEERKAT_INCOHERENTBEAMCONFIG_HPP