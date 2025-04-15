// constants.h
#pragma once

// Define mathematical constants if not already defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Physical constants (adjust as needed)
#define EPSILON_0 8.854187817e-12
#define MU_0      1.2566370614e-6
#define C_LIGHT   2.99792458e8

#define COULOMBK 1.0/(4.0 * M_PI * EPSILON_0 )  // 1 / (4 * pi * epsilon_0), or unit system-dependent

// Resonant Response Constants
#define A_PARAM 3.86237e7
#define B_PARAM 1.09367e-21
#define C_PARAM 4.56196e-37
#define F0 2.19693e14
#define GAMMA_PARAM 1.48132e13