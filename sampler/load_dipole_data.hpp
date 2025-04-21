// load_dipole_data.hpp
#ifndef LOAD_DIPOLE_DATA_HPP
#define LOAD_DIPOLE_DATA_HPP

#include <string>
#include <vector>
#include "dipole_field.hpp"

void load_dipole_data(const std::string& filename, std::vector<vec3>& positions, std::vector<cvec3>& dipoles);

#endif // LOAD_DIPOLE_DATA_HPP