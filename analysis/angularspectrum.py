import numpy as np
import matplotlib.pyplot as plt

c = 299792458
pi = np.pi

# Given a field profile which is roughly Gaussian

a = 300e-9
gwidth = 5e-6
wvl = 1e-6; freq = c/wvl; omega = 2*pi*freq
Ex = np.zeros((100,100))
Ey = np.zeros((100,100))

for x_i in range(100):
    for y_i in range(100):
        x = a * (x_i - 50)
        y = a * (y_i - 50)

        r2 = x**2 + y**2
        Ex[y_i, x_i] = np.exp(-r2/gwidth**2)
        Ey[y_i, x_i] = 0

plt.imshow(Ex)
plt.show()