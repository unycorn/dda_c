// dipole_field.hpp
#ifndef DIPOLE_FIELD_HPP
#define DIPOLE_FIELD_HPP

#include <cuComplex.h>
#include "../include/vector3.hpp"

// Custom type for complex vector
struct cvec3 {
    cuDoubleComplex x, y, z;
};

// Kernel declaration
__global__ void compute_field(
    const vec3* dipole_pos, const cvec3* dipole_mom, int N_dip,
    const vec3* obs_pos, cvec3* E_out, cvec3* B_out, int N_obs,
    double k, double prefac);

#endif // DIPOLE_FIELD_HPP
