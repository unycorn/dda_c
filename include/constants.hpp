// constants.h
#pragma once

// Define mathematical constants if not already defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Physical constants (adjust as needed)
#define EPSILON_0 8.854187817e-12
#define MU_0      1.2566370614e-6
#define Z_0       376.730313412
#define C_LIGHT   2.99792458e8

#define COULOMBK 1.0/(4.0 * M_PI * EPSILON_0 )  // 1 / (4 * pi * epsilon_0), or unit system-dependent

// Resonant Response Constants
#define EEXX_A_PARAM 3.91e7
#define EEXX_B_PARAM 6.41e-22
#define EEXX_C_PARAM 2.96e-36
#define EEXX_F0 219.8e12
#define EEXX_GAMMA_PARAM 15.0e12

#define EMXZ_A_PARAM 5.59e-8
#define EMXZ_B_PARAM 7.75e-24
#define EMXZ_C_PARAM -9.18e-39
#define EMXZ_F0 220.3e12
#define EMXZ_GAMMA_PARAM 15.2e12

#define MEZX_A_PARAM 4.42e-2
#define MEZX_B_PARAM 4.65e-17
#define MEZX_C_PARAM -2.43e-31
#define MEZX_F0 219.9e12
#define MEZX_GAMMA_PARAM 15.2e12

#define MMZZ_A_PARAM 8.15e-23
#define MMZZ_B_PARAM -5.95e-23
#define MMZZ_C_PARAM 1.48e-37
#define MMZZ_F0 220.3e12
#define MMZZ_GAMMA_PARAM 15.3e12