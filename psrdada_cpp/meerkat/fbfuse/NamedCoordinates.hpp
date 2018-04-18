#ifndef PSRDADA_CPP_MEERKAT_NAMEDCOORDINATES_HPP
#define PSRDADA_CPP_MEERKAT_NAMEDCOORDINATES_HPP

#include "psrdada_cpp/meerkat/coordinates.hpp"

template <typename CoordinatesType>
class NamedCoordinates
{
private:
    std::string _name;
    CoordinatesType _coordinates;

public:
    NamedCoordinates();
    NamedCoordinates(std::string name, CoordinatesType _coordinates);
    ~NamedCoordinates();

    std::string const& name() const;
    void name(std::string const& _name);

    CoordinatesType const& coordinates() const;
    void coordinates(CoordinatesType const& _coordinates);
};

#endif //PSRDADA_CPP_MEERKAT_NAMEDCOORDINATES_HPP