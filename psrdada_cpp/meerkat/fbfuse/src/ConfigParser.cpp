#include "psrdada_cpp/meerkat/fbfuse/ConfigParser.hpp"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>
#include <string>
#include <set>
#include <exception>
#include <iostream>
namespace pt = boost::property_tree;

namespace detail
{
    EcefCoordinates get_ecef_coords(pt::ptree const& tree)
    {
        auto parser = [&](std::string coord)->float
        {
            std::string attr(coord);
            if (tree.get<std::string>(attr.append(".<xmlattr>.units")) != "m")
                throw std::runtime_error("Units are not in meters");
            return tree.get<float>(coord);
        };
        EcefCoordinates coords;
        coords.x = parser("x");
        coords.y = parser("y");
        coords.z = parser("z");
        return coords;
    }

    RaDecCoordinates get_radec_coords(pt::ptree const& tree)
    {
        auto parser = [&](std::string coord)->float
        {
            std::string attr(coord);
            if (tree.get<std::string>(attr.append(".<xmlattr>.units")) != "rad")
                throw std::runtime_error("Units are not in radians");
            return tree.get<float>(coord);
        };
        RaDecCoordinates coords;
        coords.x = parser("x");
        coords.y = parser("y");
        return coords;
    }
} //detail

void parse_xml_config(std::string config_filename, Config& config)
{
    pt::ptree tree;
    pt::read_xml(config_filename, tree);

    //Parse subarray information
    auto& subarray = config.subarray_config();
    auto& subarray_config = tree.get_child("config.subarray");
    subarray.phase_reference(detail::get_ecef_coords(subarray_config.get_child("phase_reference")));
    subarray.boresight(detail::get_radec_coords(subarray_config.get_child("boresight_coordinates")));
    subarray.centre_frequency(subarray_config.get<float>("centre_frequency"));
    subarray.bandwidth(subarray_config.get<float>("bandwidth"));
    subarray.nchannels(subarray_config.get<std::size_t>("nchannels"));
    for(auto& antenna: subarray_config.get_child("antennas")) {
        std::string name = antenna.second.get<std::string>("name");
        EcefCoordinates coords = detail::get_ecef_coords(antenna.second);
        subarray.add_antenna(name, coords);
    }

    //Parse coherent beam information
    if (tree.count("config.output_product.coherent_beams")=0)
    {
        config.coherent_beam_config().enabled(true);
        auto& coherent_beams = config.coherent_beams_config();
        auto& coherent_beam_config = tree.get_child("config.output_product.coherent_beams");

        //beams
        for(auto& beam: coherent_beam_config.get_child("beams")) {
            std::string id = antenna.second.get<std::string>("<xmlattr>.id");
            RaDecCoordinates coords = detail::get_radec_coords(beam.second);
            coherent_beams.add_beam(id, coords);
        }

        //antennas
        for(auto& refantenna: coherent_beam_config.get_child("antennas")) {
            std::string id = refantenna.second.get<std::string>("<xmlattr>.id");
            coherent_beams.add_antenna_id(id);
        }

        coherent_beams.taccumulate(coherent_beam_config.get<std::size_t>("taccumulate"));
        coherent_beams.faccumulate(coherent_beam_config.get<std::size_t>("faccumulate"));
    }
    else
    {
        config.coherent_beams_config().enabled(false);
    }

    //Parse incoherent beam information
    if (tree.count("config.output_product.incoherent_beam")=0)
    {
        config.incoherent_beam_config().enabled(true);
        auto& incoherent_beam = config.incoherent_beam_config();
        auto& incoherent_beam_config = tree.get_child("config.output_product.incoherent_beams");

        //antennas
        for(auto& refantenna: incoherent_beam_config.get_child("antennas")) {
            std::string id = refantenna.second.get<std::string>("<xmlattr>.id");
            incoherent_beams.add_antenna_id(id);
        }

        incoherent_beams.taccumulate(incoherent_beam_config.get<std::size_t>("taccumulate"));
        incoherent_beams.faccumulate(incoherent_beam_config.get<std::size_t>("faccumulate"));
    }
    else
    {
        config.incoherent_beams_config().enabled(false);
    }


};