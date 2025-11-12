import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, mu_0, epsilon_0, pi
# from numba import jit

def gaussian_beam(x, y, z, w0, freq):
    """Gaussian beam of radius w0 and propagating along the positive x-direction"""

    wvl = c / freq
    k = 2 * np.pi / wvl
    omega = c * k

    w0 = 10e-6
    f = 1 / (k * w0)
    l = k * w0**2

    eikz = np.exp(1j * k * z)
    izlfactor = (1 + 1j*z/l)

    r = np.sqrt(x**2 + y**2)

    Ex = eikz/izlfactor * np.exp(-((r)**2) / (2 * (w0)**2 * izlfactor))
    By = Ex / c
    return Ex, By

def gaussian_beam_downward(x,y,z,w0,freq):
    """Gives the z-mirrored beam by making the transformations z -> -z and B -> -B"""
    Ex, By = gaussian_beam(x, y, -z, w0, freq)
    return Ex, -By

z = -0.25e-6
print(gaussian_beam_downward(0, 0, z, 5e-6, 300e12))