import numpy as np
import matplotlib.pyplot as plt
import re

def extract_parameters(folder_path):
    """
    Extract l, p, o, m parameters from folder path string
    Expected format: ...l0_p4_o0_m0...
    """
    # Use regex to find the pattern l<number>_p<number>_o<number>_m<number>
    pattern = r'l(\d+)_p(\d+)_o(\d+)_m(\d+)'
    match = re.search(pattern, folder_path)
    
    if match:
        l_val = int(match.group(1))
        p_val = int(match.group(2))
        o_val = int(match.group(3))
        m_val = int(match.group(4))
        return l_val, p_val, o_val, m_val
    else:
        return None, None, None, None


npz_file = "/Users/dharper/folder_data.npz"
data = np.load(npz_file, allow_pickle=True)
lst = data.files
folder_paths = data["folder_paths"].tolist()
results_x = data["results_x"]

# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = plt.cm.viridis(np.linspace(0, 1, 9))  # Use a colormap for distinct
p_dis_levels = [0, 25e-9, 50e-9, 75e-9, 100e-9, 200e-9, 300e-9, 400e-9, 500e-9]
seen_p = []
for i, folder_path in enumerate(folder_paths):
    # Extract parameters
    l_val, p_val, o_val, m_val = extract_parameters(folder_path)

    if o_val == 0:
        print(f"Folder {i}: {folder_path}")

        freq_list = results_x[i,:,0]
        r = results_x[i,:,1].astype(np.complex128)
        t = results_x[i,:,2].astype(np.complex128)

        R = np.abs(r)**2
        T = np.abs(t)**2
        A = 1 - R - T

        # plt.plot(freq_list, T, label=folder_path.split("/")[-2], color=colors[p_val % 5])
        label = None
        if p_val not in seen_p:
            seen_p.append(p_val)
            label = f"p={p_dis_levels[p_val]*1e9:.0f}nm"

        plt.plot(freq_list, np.real(r), color=colors[p_val % len(colors)], label=label)
        plt.plot(freq_list, np.imag(r), color=colors[p_val % len(colors)], linestyle='dashed')
plt.legend()
plt.show()