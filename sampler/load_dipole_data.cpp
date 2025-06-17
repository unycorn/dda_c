// load_dipole_data.cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include "dipole_field.hpp"

void load_dipole_data(const std::string& angle_filename, const std::string& dipole_filename, std::vector<vec3>& positions, std::vector<cvec3>& electric_dipoles, std::vector<cvec3>& magnetic_dipoles) {
    std::ifstream polarization_file(dipole_filename);
    std::ifstream angle_file(angle_filename);
    if (!polarization_file.is_open()) {
        std::cerr << "Error: could not open polarization_file " << dipole_filename << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (!angle_file.is_open()) {
        std::cerr << "Error: could not open angle_file " << angle_filename << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::string polarization_line;
    std::string angle_line;
    
    // Skip headers
    std::getline(polarization_file, polarization_line);
    std::getline(angle_file, angle_line);

    while (std::getline(polarization_file, polarization_line) && std::getline(angle_file, angle_line)) {
        // Process polarization file
        std::istringstream pol_ss(polarization_line);
        std::string token;
        std::vector<double> pol_values;

        while (std::getline(pol_ss, token, ',')) {
            pol_values.push_back(std::stod(token));
        }

        if (pol_values.size() < 9) {
            std::cerr << "Warning: skipping malformed polarization line\n";
            continue;
        }

        // Process angle file
        std::istringstream angle_ss(angle_line);
        std::vector<double> angle_values;
        
        while (std::getline(angle_ss, token, ',')) {
            angle_values.push_back(std::stod(token));
        }

        if (angle_values.size() < 24) {
            std::cerr << "Warning: skipping malformed angle line (expected 24 columns)\n";
            continue;
        }

        // Create position vector from polarization file
        vec3 r;
        r.x = angle_values[0];
        r.y = angle_values[1];
        r.z = angle_values[2];

        double theta = angle_values[3];

        // Create electric dipole vector from polarization file
        auto p_magnitude = make_cuDoubleComplex(pol_values[0], pol_values[1]);
        
        cvec3 p;
        p.x = cuCmul(p_magnitude, make_cuDoubleComplex(cos(theta), 0.0));
        p.y = cuCmul(p_magnitude, make_cuDoubleComplex(sin(theta), 0.0));
        p.z = make_cuDoubleComplex(0.0, 0.0);

        auto m_magnitude = make_cuDoubleComplex(pol_values[2], pol_values[3]);

        cvec3 m;
        m.x = make_cuDoubleComplex(0.0, 0.0);
        m.y = make_cuDoubleComplex(0.0, 0.0);
        m.z = cuCmul(m_magnitude, make_cuDoubleComplex(1.0, 0.0));
        
        // Store the vectors
        electric_dipoles.push_back(p);
        magnetic_dipoles.push_back(m);
        positions.push_back(r);
    }

    polarization_file.close();
    angle_file.close();
}
