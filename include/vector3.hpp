// vector3.h
#pragma once
#include <complex.h>

typedef struct {
    double x, y, z;
} vec3;

vec3 vec3_sub(vec3 a, vec3 b);
double vec3_norm(vec3 v);
vec3 vec3_scale(vec3 v, double s);
vec3 vec3_unit(vec3 v);
void outer_product(double complex out[3][3], vec3 a, vec3 b);
