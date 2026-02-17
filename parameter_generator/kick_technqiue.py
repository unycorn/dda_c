import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import time

# ----------------------------
# Parameters
# ----------------------------
N = 100
a = 300e-9
sigma = 100e-9
dmin = 100e-9
iterations = 10

# ----------------------------
# Build lattice
# ----------------------------
xs = np.arange(N) * a
ys = np.arange(N) * a
X, Y = np.meshgrid(xs, ys)
points = np.column_stack((X.ravel(), Y.ravel()))
points += np.random.normal(scale=sigma, size=points.shape)
initial_points = points.copy()


# ----------------------------
# Relaxation
# ----------------------------
def relax_until_done(points, dmin, step=1, max_iter=10):
    for it in range(max_iter):
        tree = cKDTree(points)
        pairs = tree.query_pairs(r=dmin)

        if not pairs:
            print(f"Converged after {it} iterations.")
            return points

        for i, j in pairs:
            rij = points[i] - points[j]
            dist = np.linalg.norm(rij)

            if dist == 0:
                direction = np.random.randn(2)
                direction /= np.linalg.norm(direction)
                dist = 1e-12
            else:
                direction = rij / dist

            overlap = dmin - dist
            disp = 0.5 * step * overlap * direction

            points[i] += disp
            points[j] -= disp

    print("Warning: reached max_iter without full convergence.")
    return points


def nearest_neighbor_distances(points):
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=2)
    return distances[:, 1]

# ----------------------------
# Initial distribution
# ----------------------------
initial_nn = nearest_neighbor_distances(points)

# ----------------------------
# Timing + Relaxation
# ----------------------------
start = time.perf_counter()
points = relax_until_done(points, dmin)
end = time.perf_counter()

print(f"Total relaxation time: {end - start:.6f} s")


# ----------------------------
# Final distribution
# ----------------------------
final_nn = nearest_neighbor_distances(points)

# ----------------------------
# Plot comparison
# ----------------------------
plt.figure(figsize=(6,5))

bins = 60
xrange = (0, 5)

plt.hist(initial_nn, bins=bins, range=xrange,
         alpha=0.6, label="Initial")

plt.hist(final_nn, bins=bins, range=xrange,
         alpha=0.6, label="Final")

plt.xlabel("Nearest-neighbor distance")
plt.ylabel("Count")
plt.xlim(xrange)
plt.legend()
plt.title("NN Distance: Before vs After Relaxation")

plt.show()

# ----------------------------
# Timing output
# ----------------------------
total_time = end - start
print(f"Total relaxation time: {total_time:.6f} s")
print(f"Time per iteration: {total_time/iterations:.6e} s")

from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

def get_bad_mask(points, dmin):
    tree = cKDTree(points)
    pairs = tree.query_pairs(r=dmin)
    bad = np.zeros(len(points), dtype=bool)
    for i, j in pairs:
        bad[i] = True
        bad[j] = True
    return bad

# Compute violation masks
bad_initial = get_bad_mask(initial_points, dmin)
bad_final = get_bad_mask(points, dmin)

# Plot
fig2, (axL, axR) = plt.subplots(
    1, 2, figsize=(10,5),
    sharex=True, sharey=True
)

for ax, pts, bad_mask, title in [
    (axL, initial_points, bad_initial, "Initial"),
    (axR, points, bad_final, "Final")
]:

    ax.set_title(title)
    ax.set_aspect('equal')

    # Exclusion discs
    circles = [Circle(p, dmin/2) for p in pts]
    collection = PatchCollection(
        circles,
        facecolor='none',
        edgecolor='gray',
        linewidth=0.4
    )
    ax.add_collection(collection)

    # Good points
    ax.scatter(pts[~bad_mask,0], pts[~bad_mask,1], s=20)

    # Bad points
    ax.scatter(pts[bad_mask,0], pts[bad_mask,1],
               s=40, color='red')

# Set shared limits
xmin = min(initial_points[:,0].min(), points[:,0].min()) - 1
xmax = max(initial_points[:,0].max(), points[:,0].max()) + 1
ymin = min(initial_points[:,1].min(), points[:,1].min()) - 1
ymax = max(initial_points[:,1].max(), points[:,1].max()) + 1

axL.set_xlim(xmin, xmax)
axL.set_ylim(ymin, ymax)

plt.tight_layout()
plt.show()
