#=======================================
#  Uso de GMRES para una matriz n x n
#=======================================
#  Julián T. Sagredo
#  ESFM IPN Diciembre 2025
#=======================================
from scipy.sparse.linalg import gmres
import numpy as np
import time

#==============
verificar = 0
n = 5500 
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
# Solución del sistema Ax = b usando GMRES
#============================================
t1 = time.time()
x, info = gmres(A, b, restart=n, tol=1.0e-3)
t2 = time.time()
print("GMRES (de scipy) toma: ",t2-t1)

if info == 0:
    print("GMRES converge con éxito")
    #print("Solution x:", x)
else:
    print(f"GMRES no converge después de {info} iteraciones")
    #print("Approximate solution x:", x)

# Verify the solution
if verificar == 1:
  print("A@x - b:", A@x )
  print("b:", b)

