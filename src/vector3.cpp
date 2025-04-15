// vector3.cpp

#include "vector3.hpp"
#include <cmath>
#include <complex>

// Subtract two vectors: a - b
vec3 vec3_sub(vec3 a, vec3 b) {
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}

// Norm (magnitude)
double vec3_norm(const vec3& v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

// Scale a vector
vec3 vec3_scale(const vec3& v, double s) {
    return { v.x * s, v.y * s, v.z * s };
}

// Normalize
vec3 vec3_unit(const vec3& v) {
    return vec3_scale(v, 1.0 / vec3_norm(v));
}

// Outer product of two vectors (returns 3x3 matrix)
void outer_product(std::complex<double> out[3][3], const vec3& a, const vec3& b) {
    out[0][0] = a.x * b.x;
    out[0][1] = a.x * b.y;
    out[0][2] = a.x * b.z;

    out[1][0] = a.y * b.x;
    out[1][1] = a.y * b.y;
    out[1][2] = a.y * b.z;

    out[2][0] = a.z * b.x;
    out[2][1] = a.z * b.y;
    out[2][2] = a.z * b.z;
}
