// dipole_field.hpp
#ifndef DIPOLE_FIELD_HPP
#define DIPOLE_FIELD_HPP

#include <cuComplex.h>

// Custom types
struct vec3 {
    double x, y, z;
};

struct cvec3 {
    cuDoubleComplex x, y, z;
};

// Kernel declaration
__global__ void compute_field(
    const vec3* dipole_pos, const cvec3* dipole_mom, int N_dip,
    const vec3* obs_pos, cvec3* E_out, cvec3* B_out, int N_obs,
    double k, double prefac);

#endif // DIPOLE_FIELD_HPP
