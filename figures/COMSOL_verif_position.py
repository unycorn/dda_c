import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# First dataset (100 nm displacement)
data1 = np.array([
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

# Second dataset (ideal square lattice)
data2 = np.array([
    [2.5e+14,0.00870698,0.97632],
    [2.52041e+14,0.00933221,0.97441],
    [2.54082e+14,0.0100189,0.972381],
    [2.56122e+14,0.0107773,0.970182],
    [2.58163e+14,0.0116202,0.967722],
    [2.60204e+14,0.0125615,0.964894],
    [2.62245e+14,0.0136161,0.961619],
    [2.64286e+14,0.0147992,0.957865],
    [2.66327e+14,0.0161286,0.953629],
    [2.68367e+14,0.0176262,0.948896],
    [2.70408e+14,0.0193213,0.943594],
    [2.72449e+14,0.021252,0.937565],
    [2.7449e+14,0.0234671,0.930565],
    [2.76531e+14,0.026024,0.922329],
    [2.78571e+14,0.028987,0.912647],
    [2.80612e+14,0.032431,0.901342],
    [2.82653e+14,0.0364497,0.888146],
    [2.84694e+14,0.0411644,0.8726],
    [2.86735e+14,0.046723,0.854074],
    [2.88776e+14,0.0532887,0.83192],
    [2.90816e+14,0.0610306,0.805552],
    [2.92857e+14,0.0701139,0.774338],
    [2.94898e+14,0.0806559,0.737713],
    [2.96939e+14,0.0926393,0.695544],
    [2.9898e+14,0.105781,0.648586],
    [3.0102e+14,0.119357,0.599088],
    [3.03061e+14,0.132062,0.551367],
    [3.05102e+14,0.142046,0.51179],
    [3.07143e+14,0.147343,0.487465],
    [3.09184e+14,0.146645,0.483565],
    [3.11224e+14,0.139976,0.50079],
    [3.13265e+14,0.128706,0.535002],
    [3.15306e+14,0.114882,0.579379],
    [3.17347e+14,0.100412,0.627251],
    [3.19388e+14,0.0866246,0.673795],
    [3.21429e+14,0.0742251,0.716302],
    [3.23469e+14,0.0634583,0.753685],
    [3.2551e+14,0.0542984,0.785845],
    [3.27551e+14,0.0465901,0.813185],
    [3.29592e+14,0.0401339,0.8363],
    [3.31633e+14,0.0347296,0.855819],
    [3.33673e+14,0.0301965,0.872332],
    [3.35714e+14,0.0263794,0.886354],
    [3.37755e+14,0.0231491,0.898323],
    [3.39796e+14,0.0204006,0.908599],
    [3.41837e+14,0.0180491,0.917468],
    [3.43878e+14,0.0160272,0.925158],
    [3.45918e+14,0.0142805,0.93185],
    [3.47959e+14,0.0127653,0.93769],
    [3.5e+14,0.0114458,0.942802],
])

# Process both datasets
frequency1, reflection1, transmission1 = data1[:, 0], data1[:, 1], data1[:, 2]
frequency2, reflection2, transmission2 = data2[:, 0], data2[:, 1], data2[:, 2]

# Fit cubic Bézier curves for both datasets
reflection_spline1 = CubicSpline(frequency1, reflection1)
transmission_spline1 = CubicSpline(frequency1, transmission1)
reflection_spline2 = CubicSpline(frequency2, reflection2)
transmission_spline2 = CubicSpline(frequency2, transmission2)

# Generate smooth data for plotting
frequency_smooth1 = np.linspace(frequency1.min(), frequency1.max(), 500)
frequency_smooth2 = np.linspace(frequency2.min(), frequency2.max(), 500)
reflection_smooth1 = reflection_spline1(frequency_smooth1)
transmission_smooth1 = transmission_spline1(frequency_smooth1)
reflection_smooth2 = reflection_spline2(frequency_smooth2)
transmission_smooth2 = transmission_spline2(frequency_smooth2)

# Plotting
plt.figure(figsize=(10, 6))

# Plot first dataset
line1, = plt.plot(frequency1, reflection1, 'o', label='Reflection (100 nm displacement)', markersize=4)
line2, = plt.plot(frequency1, transmission1, 'o', label='Transmission (100 nm displacement)', markersize=4)
plt.plot(frequency_smooth1, reflection_smooth1, '-', color=line1.get_color(), label='_nolegend_')
plt.plot(frequency_smooth1, transmission_smooth1, '-', color=line2.get_color(), label='_nolegend_')

# Plot second dataset
line3, = plt.plot(frequency2, reflection2, 'o', label='Reflection (ideal square lattice)', markersize=4)
line4, = plt.plot(frequency2, transmission2, 'o', label='Transmission (ideal square lattice)', markersize=4)
plt.plot(frequency_smooth2, reflection_smooth2, '-', color=line3.get_color(), label='_nolegend_')
plt.plot(frequency_smooth2, transmission_smooth2, '-', color=line4.get_color(), label='_nolegend_')

plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Reflection and Transmission vs Frequency')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()