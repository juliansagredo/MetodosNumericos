#=======================================
#  Uso de AMG para una matriz n x n
#=======================================
#  Julián T. Sagredo
#  ESFM IPN Diciembre 2025
#=======================================
#from scipy.sparse.linalg import gmres
import numpy as np
import pyamg
import time

#==============
verificar = 0
n = 1000 
#==============

#========================
#  Llenado de la matriz
#========================
print("Matriz de n x n con n = ",n)
A = np.zeros((n,n),dtype=np.float64)
b = np.zeros((n,1),dtype=np.float64)
A[:] = 1.0+np.random.randn(*A.shape)
b[:] = 1.0+np.random.randn(*b.shape)

    
#============================================
# Solución del sistema Ax = b usando AMG 
#============================================
t1 = time.time()
x = x = pyamg.solve(A, b, verb=False)
t2 = time.time()
print("AMG toma: ",t2-t1)

# Verify the solution
if verificar == 1:
  print("A @ x - b:", A @ x - b)

