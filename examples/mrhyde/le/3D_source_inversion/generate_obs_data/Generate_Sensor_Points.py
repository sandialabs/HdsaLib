import numpy as np

# Desired spacing
h = 0.3

# Domain limits
Lx, Ly, Lz = 30.0, 12.0, 3.0

def make_axis_points(L, h):
    """
    Create points from 0 to L with approximate spacing h,
    including both endpoints exactly.
    """
    n = int(round(L / h))
    return np.linspace(0.0, L, n + 1)

# Coordinate arrays
x = make_axis_points(Lx, h)
y = make_axis_points(Ly, h)
z = make_axis_points(Lz, h)

faces = []

# z = 0 and z = Lz faces
X, Y = np.meshgrid(x, y, indexing='ij')
faces.append(np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)]))
faces.append(np.column_stack([X.ravel(), Y.ravel(), Lz * np.ones(X.size)]))

# y = 0 and y = Ly faces
X, Z = np.meshgrid(x, z, indexing='ij')
faces.append(np.column_stack([X.ravel(), np.zeros(X.size), Z.ravel()]))
faces.append(np.column_stack([X.ravel(), Ly * np.ones(X.size), Z.ravel()]))

# x = 0 and x = Lx faces
Y, Z = np.meshgrid(y, z, indexing='ij')
faces.append(np.column_stack([np.zeros(Y.size), Y.ravel(), Z.ravel()]))
faces.append(np.column_stack([Lx * np.ones(Y.size), Y.ravel(), Z.ravel()]))

# Stack all face points
points = np.vstack(faces)

# Remove duplicate points along edges/corners
points = np.unique(np.round(points, decimals=12), axis=0)

# Write to file
np.savetxt("sensor_points.dat", points, fmt="%.8f")

print(f"Generated {len(points)} unique boundary points.")
