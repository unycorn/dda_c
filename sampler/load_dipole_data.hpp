// load_dipole_data.hpp
#ifndef LOAD_DIPOLE_DATA_HPP
#define LOAD_DIPOLE_DATA_HPP

#include <string>
#include <vector>
#include "dipole_field.hpp"

void load_dipole_data(
    const std::string& angle_filename, 
    const std::string& dipole_filename, 
    std::vector<vec3>& positions, 
    std::vector<cvec3>& electric_dipoles, 
    std::vector<cvec3>& magnetic_dipoles,
    double& frequency
);

#endif // LOAD_DIPOLE_DATA_HPP