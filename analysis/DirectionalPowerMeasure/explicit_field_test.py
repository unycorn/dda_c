import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, mu_0, epsilon_0, pi
import read_polarizations
# import dipole
from numba import jit

# Constants for JIT functions
MU_0 = 1.25663706127e-6
EPSILON_0 = 8.8541878188e-12
Z_0 = c * mu_0

# Add these functions here
@jit(nopython=True)
def green_E_E_dipole(r_j, r_k, k, farfield=False):
    """Electric field from electric dipole - returns 3x3 Green's tensor"""
    r = r_j - r_k
    r_len = np.sqrt(np.sum(r**2))
    
    if r_len == 0:
        return np.zeros((3, 3), dtype=np.complex128)
    
    r_hat = r / r_len
    expikr = np.exp(1j * k * r_len)
    prefac = 1.0 / (4 * np.pi * EPSILON_0) * expikr / r_len
    
    term1 = k * k
    if farfield == True:
        term2 = 0
    else:
        term2 = (1.0 - 1j * k * r_len) / (r_len * r_len)
    
    # Outer product r_hat ⊗ r_hat
    dyad = np.outer(r_hat, r_hat)
    
    # Identity matrix
    identity = np.eye(3)
    
    out = np.zeros((3, 3), dtype=np.complex128)
    for i in range(3):
        for j in range(3):
            out[i, j] = prefac * (term2 * (3.0 * dyad[i, j] - identity[i, j]) + 
                                 term1 * (identity[i, j] - dyad[i, j]))
    
    return out

@jit(nopython=True)
def farfield_E_E_dipole(px, r_source, r_sample, k):
    """Calculate far-field electric field from electric dipole"""
    # Create electric dipole moment vector
    p = np.array([px, 0.0+0.0j, 0.0+0.0j])
    
    # Get far-field Green's tensor
    G_EE = green_E_E_dipole(r_sample, r_source, k, farfield=True)
    
    # Calculate E-field: E = G_EE·p
    E_field = np.zeros(3, dtype=np.complex128)
    for i in range(3):
        for j in range(3):
            E_field[i] += G_EE[i, j] * p[j]
    
    return E_field

@jit(nopython=True)
def green_H_E_dipole(r_j, r_k, k):
    """Magnetic field from electric dipole - returns 3x3 Green's tensor"""
    r = r_j - r_k
    r_len = np.sqrt(np.sum(r**2))
    
    if r_len == 0:
        return np.zeros((3, 3), dtype=np.complex128)
    
    expikr = np.exp(1j * k * r_len)
    c = 1 / np.sqrt(EPSILON_0 * MU_0)
    omega = k * c
    prefac = -1j * omega * expikr / (4 * np.pi * r_len * r_len)
    term = (1.0 / r_len - 1j * k)
    
    # Cross product matrix [r×] where [r×]v = r × v
    cross_matrix = np.array([
        [0.0, -r[2], r[1]],
        [r[2], 0.0, -r[0]], 
        [-r[1], r[0], 0.0]
    ], dtype=np.complex128)
    
    out = np.zeros((3, 3), dtype=np.complex128)
    for i in range(3):
        for j in range(3):
            out[i, j] = -prefac * term * cross_matrix[i, j]
    
    return out

@jit(nopython=True)
def green_E_M_dipole(r_j, r_k, k):
    """Electric field from magnetic dipole - returns 3x3 Green's tensor"""
    r = r_j - r_k
    r_len = np.sqrt(np.sum(r**2))
    
    if r_len == 0:
        return np.zeros((3, 3), dtype=np.complex128)
    
    expikr = np.exp(1j * k * r_len)
    c = 1 / np.sqrt(EPSILON_0 * MU_0)
    omega = k * c
    prefac = 1j * omega * MU_0 * expikr / (4 * np.pi * r_len * r_len)
    term = (1.0 / r_len - 1j * k)
    
    # Cross product matrix [r×] where [r×]v = r × v
    cross_matrix = np.array([
        [0.0, -r[2], r[1]],
        [r[2], 0.0, -r[0]],
        [-r[1], r[0], 0.0]
    ], dtype=np.complex128)
    
    out = np.zeros((3, 3), dtype=np.complex128)
    for i in range(3):
        for j in range(3):
            out[i, j] = -prefac * term * cross_matrix[i, j]
    
    return out

@jit(nopython=True)
def green_H_M_dipole(r_j, r_k, k):
    """Magnetic field from magnetic dipole - returns 3x3 Green's tensor"""
    r = r_j - r_k
    r_len = np.sqrt(np.sum(r**2))
    
    if r_len == 0:
        return np.zeros((3, 3), dtype=np.complex128)
    
    r_hat = r / r_len
    expikr = np.exp(1j * k * r_len)
    prefac = 1.0 / (4 * np.pi) * expikr / r_len
    
    term1 = k * k
    term2 = (1.0 - 1j * k * r_len) / (r_len * r_len)
    
    # Outer product r_hat ⊗ r_hat
    dyad = np.outer(r_hat, r_hat)
    
    # Identity matrix
    identity = np.eye(3)
    
    out = np.zeros((3, 3), dtype=np.complex128)
    for i in range(3):
        for j in range(3):
            out[i, j] = prefac * (term2 * (3.0 * dyad[i, j] - identity[i, j]) + 
                                 term1 * (identity[i, j] - dyad[i, j]))
    
    return out

@jit(nopython=True)
def calculate_dipole_fields_correct(r_dipole, r_obs, px, mz, omega):
    """Calculate fields using the correct Green's functions"""
    c = 1 / np.sqrt(EPSILON_0 * MU_0)
    k = omega / c
    
    # Electric dipole moment (px, 0, 0)
    p = np.array([px, 0.0+0.0j, 0.0+0.0j])
    
    # Magnetic dipole moment (0, 0, mz)
    m = np.array([0.0+0.0j, 0.0+0.0j, mz])
    
    # Get Green's function matrices
    G_EE = green_E_E_dipole(r_obs, r_dipole, k)
    G_HE = green_H_E_dipole(r_obs, r_dipole, k)
    G_EM = green_E_M_dipole(r_obs, r_dipole, k)
    G_HM = green_H_M_dipole(r_obs, r_dipole, k)
    
    # Calculate fields: E = G_EE·p + G_EM·m, H = G_HE·p + G_HM·m
    E_total = np.zeros(3, dtype=np.complex128)
    H_total = np.zeros(3, dtype=np.complex128)
    
    for i in range(3):
        for j in range(3):
            E_total[i] += G_EE[i, j] * p[j] + G_EM[i, j] * m[j]
            H_total[i] += G_HE[i, j] * p[j] + G_HM[i, j] * m[j]
    
    return np.array([E_total[0], E_total[1], E_total[2]]), np.array([H_total[0], H_total[1], H_total[2]])

r_dipole_1 = np.array([0,0,0])
r_dipole_2 = np.array([100e-9,0,0])
omega = 300e12 * 2*pi

obs_R = 10e-6
total_power = 0

theta_res = 1000; phi_res = 2000

for theta in np.linspace(0, pi, theta_res):
    for phi in np.linspace(0, 2*pi, phi_res):
        delta_theta = pi / theta_res
        delta_phi = 2*pi / phi_res

        
        r_obs_unit = np.array([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)])
        r_obs = obs_R * r_obs_unit

        Evec1, Hvec1 = calculate_dipole_fields_correct(r_dipole_1, r_obs, 1e-31, 0, omega)
        Evec2, Hvec2 = calculate_dipole_fields_correct(r_dipole_2, r_obs, -1e-31, 0, omega)
        Evec = Evec1 + Evec2
        Hvec = Hvec1 + Hvec2

        Poynting_avg = 0.5 * np.real( np.cross(Evec, np.conj(Hvec)) )
        normal_Poynting_avg = np.dot(r_obs_unit, Poynting_avg)

        integral_factor = obs_R**2 * np.sin(theta) * delta_theta * delta_phi

        total_power += normal_Poynting_avg * integral_factor

print(total_power)