import pyamg
from scipy.sparse import csr_matrix
import numpy as np
import time

n = 4097 
print("Ecuación de Poisson con nodos = ",n*n)
A = pyamg.gallery.poisson((n,n), format='csr')  # 2D Poisson problem on 500x500 grid
#A = pyamg.gallery.sprand(n*n, n*n, density=0.01, format='csr')
b = np.random.rand(A.shape[0])                      # pick a random right hand side
time1 = time.time()
ml = pyamg.ruge_stuben_solver(A)                    # construct the multigrid hierarchy
print(ml)                                           # print hierarchy information
x = ml.solve(b, tol=1e-10)                          # solve Ax=b to a tolerance of 1e-10
time2 = time.time()
print("AMG tarda = ",time2-time1)
print("residual: ", np.linalg.norm(b-A*x)) 
#print("Solution x:", x)
