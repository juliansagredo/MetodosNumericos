#=================================
#  GMRES para matrices esparsas
#=================================
#  Julián T. Sagredo
#  ESFM  IPN  2025
#=================================
import scipy.sparse
import numpy as np
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from time import perf_counter_ns
import jax.numpy as jnp
import jax

np.random.seed(179)
n = 500
density = 0.01
N = int(n*n*density)
shape = (n, n)
print("Invertir matriz de n x n con n = ", n)
print("Matriz esparsa con entradas = ", N)


#=============================================
# Crear matriz esparsa de nxn con N entradas 
#=============================================
coords = np.random.choice(n * n, size=N, replace=False)
coords = np.unravel_index(coords, shape)
values = np.random.normal(size=N)
A_sparse = scipy.sparse.coo_matrix((values, coords), shape=shape)
A_sparse = A_sparse.tocsr()
A_sparse += scipy.sparse.eye(n)
A_dense = A_sparse.toarray()

b = np.random.normal(size=n)
#b = A_sparse @ b

# Solve using np.linalg.lstsq
#time_before = perf_counter_ns()
#x = np.linalg.lstsq(A_dense, b, rcond=None)[0]
#time_taken = (perf_counter_ns() - time_before) * 1e-6
#error = np.linalg.norm(A_dense @ x - b) ** 2
#print(f"Using dense solver: error: {error:.4e} in time {time_taken:.1f}ms")

# Solve using inverse matrix
#time_before = perf_counter_ns()
#x = np.linalg.inv(A_dense) @ x
#time_taken = (perf_counter_ns() - time_before) * 1e-6
#error = np.linalg.norm(A_dense @ x - b) ** 2
#print(f"Using matrix inversion: error: {error:.4e} in time {time_taken:.1f}ms")

# Solve using GMRES
time_before = perf_counter_ns()
x = scipy.sparse.linalg.gmres(A_sparse, b, tol=1e-8)[0]
time_taken = (perf_counter_ns() - time_before) * 1e-6
error = np.linalg.norm(A_sparse @ x - b) ** 2
print(f"Using sparse scipy gmres: error: {error:.4e} in time {time_taken:.1f}ms")

def gmres(linear_map, b, x0, n_iter):
    # Initialization
    n = x0.shape[0]
    H = np.zeros((n_iter + 1, n_iter))
    r0 = b - linear_map(x0)
    beta = np.linalg.norm(r0)
    V = np.zeros((n_iter + 1, n))
    V[0] = r0 / beta

    for j in range(n_iter):
        # Compute next Krylov vector
        w = linear_map(V[j])

        # Gram-Schmidt orthogonalization
        for i in range(j + 1):
            H[i, j] = np.dot(w, V[i])
            w -= H[i, j] * V[i]
        H[j + 1, j] = np.linalg.norm(w)

        # Add new vector to basis
        V[j + 1] = w / H[j + 1, j]

    # Find best approximation in the basis V
    e1 = np.zeros(n_iter + 1)
    e1[0] = beta
    y = np.linalg.lstsq(H, e1, rcond=None)[0]

    # Convert result back to full basis and return
    x_new = x0 + V[:-1].T @ y
    return x_new

# Try out the GMRES routine
time_before = perf_counter_ns()
x0 = np.zeros(n)
linear_map = lambda x: A_sparse @ x
x = gmres(linear_map, b, x0, 50)
time_taken = (perf_counter_ns() - time_before) * 1e-6
error = np.linalg.norm(A_sparse @ x - b) ** 2
print(f"Using GMRES: error: {error:.4e} in time {time_taken:.1f}ms")

