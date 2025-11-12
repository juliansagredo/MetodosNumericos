#====================================
#  Julián Sagredo
#====================================
#  Paradigmas de Programación
#  Matemática Algorítmica
#  ESFM  IPN  Noviembre 2024
#====================================

#===============================
#  Importar módulos necesarios
#===============================
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time
from numba import jit

#=========================
#  Parámetros de entrada 
#=========================
#-------------------------------------------
# Número de celdas
n = np.array([512,512])
# Tamaño del dominio (menor que uno) 
L = np.array([1.0,1.0]) 
# Constante de difusión 
k = 0.2
# Pasos de tiempo
pasos = 1000
#-------------------------------------------

#=========================
#  Parámetros secundarios 
#=========================
# Tamaño de las celdas
dx = L/n
udx2 = 1.0/(dx*dx)
# Paso de tiempo
dt = 0.25*(min(dx[0],dx[1])**2)/k
print("dt = ",dt)
# Total de celdas
nt = n[0]*n[1]
print("celdas = ",nt)

#======================
#  Arreglos iniciales 
#======================
# Llenar la solución con ceros 
u = np.zeros(nt,dtype=np.float64)   # arreglo de lectura
un = np.zeros(nt,dtype=np.float64)  # arreglo de escritura

#====================================
#  Función sin intérprete de python 
#====================================
@jit(nopython=True)
def evolucion(u,n,udx2,dt,i,k):
    jp1 = i + n[0]
    jm1 = i - n[0]
    laplaciano = (u[i-1]-2.0*u[i]+u[i+1])*udx2[0] + (u[jm1]-2.0*u[i]+u[jp1])*udx2[1]
    unueva = u[i] + dt*k*laplaciano
    return unueva

#===========================================
#  Loop acelerado sin intérprete de python
#===========================================
@jit(nopython=True)
def solucion(u,un,udx2,dt,n,k):
   for jj in range(1,n[1]-1):
     for ii in range(1,n[0]-1):
         i = ii + n[0]*jj
         unueva = evolucion(u,n,udx2,dt,i,k)
         if i == int(nt/2)+int(n[0]/2):
             unueva = 1.0
         un[i] = unueva

#======================
#  PROGRAMA PRINCIPAL 
#======================
start = time.time()
for t in range(1,pasos+1):
    solucion(u,un,udx2,dt,n,k)
    u = un
    if t%100==0: print("Iteración = ",t)
end = time.time()
print("Tardó: ",end-start,"s")

#=======================================
#  Graficar la solución al tiempo final 
#=======================================
x,y = np.meshgrid(np.arange(0,L[0],dx[0]),np.arange(0,L[1],dx[1]))
ax = plt.axes(projection='3d')
up = np.reshape(u,(n[0],n[1]))
ax.plot_surface(x,y,up,cmap=cm.hsv)
plt.show()

