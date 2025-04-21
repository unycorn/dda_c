// load_dipole_data.cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include "dipole_field.hpp"

void load_dipole_data(const std::string& filename, std::vector<vec3>& positions, std::vector<cvec3>& dipoles) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: could not open file " << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::string line;
    std::getline(file, line);  // Skip header

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        std::vector<double> values;

        while (std::getline(ss, token, ',')) {
            values.push_back(std::stod(token));
        }

        if (values.size() < 9) {
            std::cerr << "Warning: skipping malformed line\n";
            continue;
        }

        cvec3 p;
        p.x = make_cuDoubleComplex(values[0], values[1]);
        p.y = make_cuDoubleComplex(values[2], values[3]);
        p.z = make_cuDoubleComplex(values[4], values[5]);

        vec3 r;
        r.x = values[6];
        r.y = values[7];
        r.z = values[8];

        dipoles.push_back(p);
        positions.push_back(r);
    }

    file.close();
}
