import numpy as np
import matplotlib.pyplot as plt

data = np.load("analysis/gaussian_beam_patch/combined_power_data.npz")
freq = data["frequencies"]
for key in data["folder_names"]:
    P0 = data[key + "_P0"]
    Pt = data[key + "_Pt"]

    plt.plot(freq, Pt, label=key)
plt.legend()
plt.show()