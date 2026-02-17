import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

def generate_lattice_with_min_distance(N, a, sigma, dmin, max_attempts=1000):
    """
    Generate NxN square lattice with Gaussian perturbations,
    rejecting samples that violate minimum distance.

    Returns:
        ideal_positions        : (N^2, 2)
        perturbed_positions    : (N^2, 2)
    """

    ideal = np.empty((N, N, 2))
    perturbed = np.empty((N, N, 2))

    for i in range(N):
        for j in range(N):

            ideal_pos = np.array([i*a, j*a])
            ideal[i, j] = ideal_pos

            for _ in range(max_attempts):

                candidate = ideal_pos + np.random.normal(scale=sigma, size=2)

                ok = True

                # Check neighbors within ±2 lattice sites
                for di in (-2, -1, 0, 1, 2):
                    for dj in (-2, -1, 0, 1, 2):

                        ni = i + di
                        nj = j + dj

                        if 0 <= ni < N and 0 <= nj < N:

                            # Only check already placed sites
                            if ni < i or (ni == i and nj < j):

                                dist = np.linalg.norm(candidate - perturbed[ni, nj])
                                if dist < dmin:
                                    ok = False
                                    break
                    if not ok:
                        break

                if ok:
                    perturbed[i, j] = candidate
                    break

            else:
                raise RuntimeError(f"Failed to place site ({i},{j}).")

    return ideal.reshape(-1, 2), perturbed.reshape(-1, 2)


def nearest_neighbor_distances(points):
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=2)
    return distances[:, 1]

sigma = 100e-9

ideal_points, points = generate_lattice_with_min_distance(100, 300e-9, sigma, 100e-9)
_, unfixed_points = generate_lattice_with_min_distance(100, 300e-9, sigma, 0)

bad_nn = nearest_neighbor_distances(unfixed_points)
final_nn = nearest_neighbor_distances(points)

print(np.sum(bad_nn < 100e-9))
print(np.sum(final_nn < 100e-9))

# ----------------------------
# Plot comparison
# ----------------------------
plt.figure(figsize=(6,5))

bins = 60
xrange = (0, 500e-9)

plt.hist(bad_nn, bins=bins, range=xrange,
         alpha=0.6, label="Initial")

plt.hist(final_nn, bins=bins, range=xrange,
         alpha=0.6, label="Final")

plt.xlabel("Nearest-neighbor distance")
plt.ylabel("Count")
plt.xlim(xrange)
plt.legend()
plt.title("NN Distance: Before vs After Relaxation")

plt.show()