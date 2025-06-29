// load_dipole_data.cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <complex>
#include "dipole_field.hpp"
#include "../include/fileio.hpp"

void load_dipole_data(const std::string& angle_filename, const std::string& dipole_filename, std::vector<vec3>& positions, std::vector<cvec3>& electric_dipoles, std::vector<cvec3>& magnetic_dipoles) {
    std::ifstream angle_file(angle_filename);
    if (!angle_file.is_open()) {
        std::cerr << "Error: could not open angle_file " << angle_filename << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Read binary polarization data
    int N;
    double frequency;
    std::vector<std::complex<double>> pol_data = read_polarizations_binary(dipole_filename.c_str(), N, frequency);

    std::string angle_line;
    // Skip header
    std::getline(angle_file, angle_line);

    int point_idx = 0;
    while (std::getline(angle_file, angle_line) && point_idx < N) {
        // Process angle file
        std::istringstream angle_ss(angle_line);
        std::string token;
        std::vector<double> angle_values;
        
        while (std::getline(angle_ss, token, ',')) {
            angle_values.push_back(std::stod(token));
        }

        if (angle_values.size() < 24) {
            std::cerr << "Warning: skipping malformed angle line (expected 24 columns)\n";
            continue;
        }

        // Create position vector from angle file
        vec3 r;
        r.x = angle_values[0];
        r.y = angle_values[1];
        r.z = angle_values[2];

        double theta = angle_values[3];

        // Create electric dipole vector from binary data
        // pol_data[2*point_idx + 0] contains Ex
        auto p_magnitude = make_cuDoubleComplex(pol_data[2*point_idx + 0].real(), pol_data[2*point_idx + 0].imag());
        
        cvec3 p;
        p.x = cuCmul(p_magnitude, make_cuDoubleComplex(cos(theta), 0.0));
        p.y = cuCmul(p_magnitude, make_cuDoubleComplex(sin(theta), 0.0));
        p.z = make_cuDoubleComplex(0.0, 0.0);

        // pol_data[2*point_idx + 1] contains Mz
        auto m_magnitude = make_cuDoubleComplex(pol_data[2*point_idx + 1].real(), pol_data[2*point_idx + 1].imag());

        cvec3 m;
        m.x = make_cuDoubleComplex(0.0, 0.0);
        m.y = make_cuDoubleComplex(0.0, 0.0);
        m.z = cuCmul(m_magnitude, make_cuDoubleComplex(1.0, 0.0));
        
        // Store the vectors
        electric_dipoles.push_back(p);
        magnetic_dipoles.push_back(m);
        positions.push_back(r);

        point_idx++;
    }

    angle_file.close();
}
