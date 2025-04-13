// vector3.c
#include "vector3.h"
#include <math.h>

// Subtract two vectors: a - b
vec3 vec3_sub(vec3 a, vec3 b) {
    vec3 r;
    r.x = a.x - b.x;
    r.y = a.y - b.y;
    r.z = a.z - b.z;
    return r;
}

// Norm (magnitude)
double vec3_norm(vec3 v) {
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

// Scale a vector
vec3 vec3_scale(vec3 v, double s) {
    vec3 r;
    r.x = v.x * s;
    r.y = v.y * s;
    r.z = v.z * s;
    return r;
}

// Normalize
vec3 vec3_unit(vec3 v) {
    return vec3_scale(v, 1.0 / vec3_norm(v));
}

// Outer product of two vectors (returns 3x3 matrix)
void outer_product(double complex out[3][3], vec3 a, vec3 b) {
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
