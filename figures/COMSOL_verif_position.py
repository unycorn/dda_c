import numpy as np
from scipy.interpolate import CubicSpline

import matplotlib.pyplot as plt

# Data points
data = np.array([
    [2.5e+14, 0.00974144, 0.973304],
    [2.52041e+14, 0.0104801, 0.971116],
    [2.54082e+14, 0.011299, 0.968746],
    [2.56122e+14, 0.0122111, 0.966123],
    [2.58163e+14, 0.0132315, 0.963155],
    [2.60204e+14, 0.0143771, 0.959736],
    [2.62245e+14, 0.0156667, 0.95576],
    [2.64286e+14, 0.0171206, 0.951157],
    [2.66327e+14, 0.0187636, 0.945919],
    [2.68367e+14, 0.02063, 0.940034],
    [2.70408e+14, 0.0227669, 0.93338],
    [2.72449e+14, 0.025232, 0.925705],
    [2.7449e+14, 0.0280918, 0.916691],
    [2.76531e+14, 0.031421, 0.905997],
    [2.78571e+14, 0.0353043, 0.893305],
    [2.80612e+14, 0.0398455, 0.87832],
    [2.82653e+14, 0.0451842, 0.860655],
    [2.84694e+14, 0.0514973, 0.839667],
    [2.86735e+14, 0.0589694, 0.814524],
    [2.88776e+14, 0.0677557, 0.784455],
    [2.90816e+14, 0.0779516, 0.748928],
    [2.92857e+14, 0.0895279, 0.707786],
    [2.94898e+14, 0.10217, 0.66166],
    [2.96939e+14, 0.115057, 0.612781],
    [2.9898e+14, 0.126724, 0.565677],
    [3.0102e+14, 0.135259, 0.526919],
    [3.03061e+14, 0.139202, 0.502721],
    [3.05102e+14, 0.138692, 0.495257],
    [3.07143e+14, 0.135038, 0.502638],
    [3.09184e+14, 0.128713, 0.523446],
    [3.11224e+14, 0.119364, 0.556923],
    [3.13265e+14, 0.107518, 0.599483],
    [3.15306e+14, 0.0946133, 0.645492],
    [3.17347e+14, 0.0820128, 0.690184],
    [3.19388e+14, 0.0705337, 0.730831],
    [3.21429e+14, 0.0604973, 0.766399],
    [3.23469e+14, 0.0519225, 0.796865],
    [3.2551e+14, 0.0446839, 0.822679],
    [3.27551e+14, 0.0386053, 0.844454],
    [3.29592e+14, 0.033506, 0.862814],
    [3.31633e+14, 0.0292208, 0.878327],
    [3.33673e+14, 0.0256064, 0.891489],
    [3.35714e+14, 0.0225422, 0.902719],
    [3.37755e+14, 0.019929, 0.912365],
    [3.39796e+14, 0.0176866, 0.920707],
    [3.41837e+14, 0.0157511, 0.927963],
    [3.43878e+14, 0.0140723, 0.934301],
    [3.45918e+14, 0.0126105, 0.939847],
    [3.47959e+14, 0.0113334, 0.944707],
    [3.5e+14, 0.0102146, 0.948974],
])

# Extract columns
frequency = data[:, 0]
reflection = data[:, 1]
transmission = data[:, 2]

# Fit cubic Bézier curves
reflection_spline = CubicSpline(frequency, reflection)
transmission_spline = CubicSpline(frequency, transmission)

# Generate smooth data for plotting
frequency_smooth = np.linspace(frequency.min(), frequency.max(), 500)
reflection_smooth = reflection_spline(frequency_smooth)
transmission_smooth = transmission_spline(frequency_smooth)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(frequency, reflection, 'o', label='Reflection (Data)', markersize=4)
plt.plot(frequency, transmission, 'o', label='Transmission (Data)', markersize=4)
plt.plot(frequency_smooth, reflection_smooth, '-', label='Reflection (Cubic Bézier)')
plt.plot(frequency_smooth, transmission_smooth, '-', label='Transmission (Cubic Bézier)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Reflection and Transmission vs Frequency')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()