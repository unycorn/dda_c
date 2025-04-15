#ifndef VECTOR3_HPP
#define VECTOR3_HPP

#include <complex>

// Basic 3D vector
struct vec3 {
    double x, y, z;
};

// Vector operations
vec3 vec3_sub(vec3 a, vec3 b);
double vec3_norm(const vec3& v);
vec3 vec3_scale(const vec3& v, double s);
vec3 vec3_unit(const vec3& v);

// Outer product (returns a 3x3 complex matrix)
void outer_product(std::complex<double> out[3][3], const vec3& a, const vec3& b);

#endif // VECTOR3_HPP
