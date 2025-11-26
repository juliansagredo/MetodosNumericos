#===============================
#  Simple eliminación de Gauss
#===============================
# - Acelerado con numpy y numba
# - Comparado con numpy y cupy
#===============================
#  Julián T. Becerra Sagredo
#  ESFM IPN
#===============================

#===========
#  Módulos 
#===========
from numba import jit
import cupy 
import numpy as np
import sys
import time

#=====================================
#  Algoritmo de eliminación de Gauss 
#=====================================
@jit(nopython=True)
def gauss(a:np.array):
  n = a.shape[0]
  x = np.zeros(n)
  for i in range(n):
    if a[i,i] == 0.0:
        print('Contiene una división entre cero!')
        return
    for j in range(i+1, n):
        ratio:np.float64 = a[j,i]/a[i,i]
        for k in range(n+1):
            a[j,k] = a[j,k] - ratio * a[i,k]

  #====================================
  #  Sustitución de abajo para arriba 
  #====================================
  x[n-1] = a[n-1,n]/a[n-1,n-1]
  for i in range(n-2,-1,-1):
    x[i] = a[i,n]
    for j in range(i+1,n):
        x[i] = x[i] - a[i,j]*x[j]
    x[i] = x[i]/a[i,i]
  return x 

#======================
#  Programa de prueba 
#======================
if __name__=="__main__":
  #==================
  #  Dimensión n x n  
  #==================
  n = 1000
  #======================================
  #  Matriz aumentada (números al azar) 
  #======================================
  a = np.zeros((n,n+1),dtype=np.float64)
  for i in range(n):
    for j in range(n+1):
        a[i,j] = 1.0+np.random.rand(1)
  #============================================
  #  Obtener la soución con nuestro algoritmo
  #============================================
  t1 = time.time()
  x = gauss(a) 
  np.linalg.solve  
  t2 = time.time()
  print("Nuestro algoritmo toma ",t2-t1)

  #==============================
  #  Obtener solución con numpy 
  #==============================
  t1 = time.time()
  y = np.linalg.solve(a[0:n,0:n],a[0:n,n])
  t2 = time.time()
  print("Algoritmo de numpy toma ",t2-t1)

  #=============================
  #  Obtener solución con cupy 
  #=============================
  A = cupy.array(a)
  t1 = time.time()
  z = cupy.linalg.solve(A[0:n,0:n],A[0:n,n])
  t2 = time.time()
  print("Algoritmo de cupy toma ",t2-t1)

  #============================
  #  Matriz diagonal con cupy
  #============================
  #index = np.random.randint(0,n,10)
  #for i in index:
  #  print("x1-x2 = ", i,x[i]-y[i])
  #  print("x1-x3 = ", i,x[i]-z[i]) 


