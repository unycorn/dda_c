#ifndef VECTOR3_HPP
#define VECTOR3_HPP

#include <complex>
#include <stdexcept>

// Basic 3D vector
struct vec3 {
    double x, y, z;
    
    double& operator[](int i) {
        switch(i) {
            case 0: return x;
            case 1: return y;
            case 2: return z;
            default: throw std::out_of_range("vec3 index out of range");
        }
    }
    
    const double& operator[](int i) const {
        switch(i) {
            case 0: return x;
            case 1: return y;
            case 2: return z;
            default: throw std::out_of_range("vec3 index out of range");
        }
    }
};

// Vector operations
vec3 vec3_sub(vec3 a, vec3 b);
double vec3_norm(const vec3& v);
vec3 vec3_scale(const vec3& v, double s);
vec3 vec3_unit(const vec3& v);
double vec3_dot(const vec3& a, const vec3& b);  // New dot product function

// Complex vector operations
std::complex<double> vec3_dot(const vec3& a, const vec3& b_complex);  // Overload for complex vectors

// Outer product (returns a 3x3 complex matrix)
void outer_product(std::complex<double> out[3][3], const vec3& a, const vec3& b);

#endif // VECTOR3_HPP
