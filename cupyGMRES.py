#==================================
#  GMRES en cupy
#==================================
#  Julián T. Sagredo
#  ESFM IPN 2025 
#==================================
import numpy as np
import cupy as cp
import scipy.sparse as sp
import cupyx.scipy.sparse as cupy_sparse
from scipy.sparse.linalg import gmres as scipy_gmres
from cupyx.scipy.sparse.linalg import gmres as cupy_gmres
from scipy.sparse.linalg import spsolve as scipy_spsolve
from cupyx.scipy.sparse.linalg import spsolve as cupy_spsolve
import time


#==================
#  Semilla al azar
#==================
np.random.seed(2025)

#=================================
#  Generar matriz esparsa al azar 
#  Densidad < 1
#=================================
size = 5500  
print("Matriz de n x n con n = ",size)
density = 0.01 
print("Matriz esparsa con densidad = ",density)
A = sp.random(size, size, density=density, format="csr", dtype=np.float64) + 1 * sp.eye(size)

#=========================
# Generar lado derecho b
#=========================
b = np.random.randn(size)

#================
#  Pasar al GPU
#================
A_cupy = cupy_sparse.csr_matrix(A)
b_cupy = cp.array(b)

# SciPy direct solve
t1 = time.time()
x1 = scipy_spsolve(A, b)
t2 = time.time()
print("Tiempo de scipy direct = ",t2-t1)

# CuPy direct solve
t1 = time.time()
x2 = cupy_spsolve(A_cupy, b_cupy)
t2 = time.time()
print("Tiempo de cupy direct = ",t2-t1)

print('relative difference between direct solvers',
      np.linalg.norm(x1 - cp.asnumpy(x2)) / np.linalg.norm(x1))

# Solve using SciPy GMRES
t1 = time.time()
x3, _ = scipy_gmres(A, b,restart=size)
t2 = time.time()
print("Tiempo de scipy gmres = ",t2-t1)

# Solve using CuPy GMRES
t1 = time.time()
x4, _ = cupy_gmres(A_cupy, b_cupy,restart=size)
t2 = time.time()
print("Tiempo de cupy gmres = ",t2-t1)


# Compute relative difference
print('relative difference between gmres',
      np.linalg.norm(x3 - cp.asnumpy(x4)) / np.linalg.norm(x3))
