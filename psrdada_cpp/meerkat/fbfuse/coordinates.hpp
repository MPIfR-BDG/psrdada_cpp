#ifndef PSRDADA_CPP_MEERKAT_COORDINATES_HPP
#define PSRDADA_CPP_MEERKAT_COORDINATES_HPP

struct EcefCoordinates
{
    double x;
    double y;
    double z;
};

struct RaDecCoordinates
{
    double ra;
    double dec;
};

struct AzAltCoordinates
{
    double az;
    double alt;
};

#endif //PSRDADA_CPP_MEERKAT_COORDINATES_HPP