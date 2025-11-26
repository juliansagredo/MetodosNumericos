from scipy.sparse.linalg import gmres
import numpy as np
import time

# Define la matriz A y el vector b
#A = np.array([[...], [...], ...])
#b = np.array([...])
n = 100
A = np.zeros((n,n),dtype=np.float64)
b = np.zeros((n,1),dtype=np.float64)
for i in range(n):
  b[i] = 1.0+np.random.rand(1)
  for j in range(n):
    A[i,j] = 1.0+np.random.rand(1)
    

# Resuelve el sistema Ax = b usando gmres
t1 = time.time()
x, info = gmres(A, b)
t2 = time.time()
print("Algoritmo gmres toma ",t2-t1)


# Verifica el resultado
#print(f"Resultado: {x}")

import cupy as cp
import numpy as np
from pycuGMRES import pycuGMRESold

def solve_linear_system_with_pycugmres(A,b):
    """
    Solves a linear system Ax = b using pycugmres.
    """
    
    # Define the matrix A and vector b (using numpy initially)
    # A simple, small example for demonstration purposes.
    # pycugmres is designed for much larger systems.
    A_host = A 
    b_host = b 

    # Move data to the GPU using cupy
    A_gpu = cp.asarray(A_host)
    b_gpu = cp.asarray(b_host)

    # Initialize the GMRES solver instance
    # Parameters can be adjusted (e.g., tolerance, max iterations, restart value)
    solver = pycuGMRESold(tol=1e-6, max_iter=100, restart=30, verbosity=False)

    print("Solving linear system Ax = b using pycugmres on GPU...")

    # Solve the system
    # The solver operates in-place on the 'b_gpu' array to store the result 'x'
    x_gpu = solver.solve(A_gpu, b_gpu)

    # Move the solution back to the host (CPU) for verification/display
    x_host = cp.asnumpy(x_gpu)

    print("\nSolution x:")
    print(x_host)

    # Verification: Check A * x
    Ax_host = np.dot(A_host, x_host)
    print("\nVerification (A * x):")
    print(Ax_host)
    print("\nOriginal b:")
    print(b_host)

    # Check the difference
    difference = np.linalg.norm(Ax_host - b_host)
    print(f"\nNorm of difference ||Ax - b||: {difference}")

    if difference < 1e-5:
        print("Solution is accurate.")
    else:
        print("Solution may not be accurate.")

solve_linear_system_with_pycugmres(A,b)

