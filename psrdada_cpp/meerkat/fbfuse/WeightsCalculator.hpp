#ifndef PSRDADA_CPP_MEERKAT_WEIGHTSCALCULATOR_HPP
#define PSRDADA_CPP_MEERKAT_WEIGHTSCALCULATOR_HPP

#include "psrdada_cpp/meerkat/coordinates.hpp"

class WeightsCalculator
{
private:
    SubarrayConfiguration const& _subarray_config;
    BeamConfiguration const& _beam_config;
    weight_t* _weights_d; //These are 8-bit weights in FBPA order in device memory
    std::vector<AzAltCoordinates> _beam_azalt; //AzAlt beam positions in host memory
    AzAltCoordinates* _beam_azalt_d; //AzAlt beam positions in device memory
    std::vector<frequency_t> _frequencies; //Channel frequencies in host memory
    frequency_t* _frequencies_d; //Channel frequencies in device memory

    /*
    internally there will be a cuda call for the weights generation that requires the following:
    alt-az coordinates for each beam
    sky frequencies for each channel
    antenna positions in ECEF coordinates

    Weights need to be ordered appropriately when created, this means FBPA order.
    The weights calculator should specify the validity of the weights based on a maximum acceptable beam drift.
    The drift can be characterised as a S/N loss. This need only be calculated for the beam with the biggest
    offset from the boresight.

    On the CPU side this class holds responsibility for converting beam positions from equatorial to
    horizontal coordinates. This should be done using SOFA or katpoint (if there is a C++ wrapper).

    From SOFA we require iauAtco13 for conversion between ICRS J2000 RA Dec and azimuth and
    zenith angles at the epoch of observation. For this we need to provide:
        ** rc,dc double ICRS right ascension at J2000.0 (radians, Note 1)
        ** pr double RA proper motion (radians/year; Note 2)
        ** pd double Dec proper motion (radians/year)
        ** px double parallax (arcsec)
        ** rv double radial velocity (km/s, +ve if receding)
        ** utc1 double UTC as a 2−part...
        ** utc2 double ...quasi Julian Date (Notes 3−4)
        ** dut1 double UT1−UTC (seconds, Note 5)
        ** elong double longitude (radians, east +ve, Note 6)
        ** phi double latitude (geodetic, radians, Note 6)
        ** hm double height above ellipsoid (m, geodetic, Notes 6,8)
        ** xp,yp double polar motion coordinates (radians, Note 7)
        ** phpa double pressure at the observer (hPa = mB, Note 8)
        ** tc double ambient temperature at the observer (deg C)
        ** rh double relative humidity at the observer (range 0−1)
        ** wl double wavelength (micrometers, Note 9)
    The slightly tricky parameters to get are:
        - the UTC/UT1 offset -> comes from the IERS bulletin A (we can get this through astropy)
        - polar motion coordinates -> also in IERS bulletin A
        - the pressure at the observatory (CAM?)
        - the temperature at the observatory (CAM?)
        - the humidity at the observatory (CAM?)
    */


    /**
     * @brief Update the horizontal coordinates of all the beams for the current epoch
     * @details [long description]
     *
     * @param time [description]
     */
    void _update_beam_altaz(time_t time);


public:
    WeightsCalculator(SubarrayConfiguration const& subarray_config, BeamConfiguration const& beam_config);
    ~WeightsCalculator();


    /**
     * @brief Update the weights for the epoch given
     *
     * @details
     * The control flow here should be something like the following:
     * 1. Use time to calculate
     *
     * @param time [description]
     * @return [description]
     */
    weight_t const* calculate(time_t time);

};

#endif //PSRDADA_CPP_MEERKAT_WEIGHTSCALCULATOR_HPP