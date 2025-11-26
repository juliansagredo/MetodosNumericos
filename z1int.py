#=================================
#  Integral con el Z-spline 
#  cúbico para puntos arbitrarios
#=================================
#  Julián T. Sagredo
#  Septiembre 2025
#  ESFM IPN
#=================================

#=====================
#  numpy y matplotlib
#=====================
import numpy as np
import matplotlib.pyplot as plt

#======================
#  Integral Z1-spline 
#======================
def z1int(x:np.array,f:np.float64):
    n = len(x)
    w = np.zeros(n,dtype=np.float64)
    dx = np.zeros(n-1,dtype=np.float64) 
    #========================
    #  Cálculo de intervalos
    #========================
    for i in range(0,n-1):
      dx[i] = x[i+1]-x[i] 
    #===============================
    #  Coeficientes de integración
    #===============================
    for i in range(1,n-2):
      w[i] = 0.5*(dx[i]+dx[i-1]) 
    for i in range(2,n-3):
      w[i] += (dx[i]**3+dx[i-1]**3)/(12.0*dx[i]*dx[i-1]) \
            - (dx[i]**3+dx[i+1]**3)/(12.0*dx[i]*(dx[i]+dx[i+1])) \
            - (dx[i-1]**3+dx[i-2]**3)/(12.0*dx[i-1]*(dx[i-1]+dx[i-2]))
    w[0] = 0.5*dx[0]
    w[n-1] = 0.5*dx[n-2] 
    #================================
    #  Integral (producto punto f.w) 
    #================================
    y = np.dot(f,w)

    return y 

#=========
#  MAIN  
#=========
if __name__ == "__main__":
   #================================
   # Ejemplo 1: el Z1 para ai dado 
   #================================
   N = 5
   x1 = np.linspace(0.0,1.0,N)
   f1 = np.sin(np.pi*x1)
   rng = np.random.default_rng() 
   x2 = rng.uniform(low=0.0, high=1.0, size=N)
   x2 = np.sort(x2)
   x2[0] = 0.0
   x2[N-1] = 1.0
   f2 = np.sin(np.pi*x2) 
   intf1 = z1int(x1,f1) 
   intf2 = z1int(x2,f2)
   print("Error de la integral (equidistante) = ",intf1-2.0/np.pi)
   print("Error de la integral (al azar) = ",intf2-2.0/np.pi)
   plt.plot(x2,f2,linewidth=4,color="purple")
   plt.plot(x1,f1,linewidth=1,color="orange")
   plt.grid()
   plt.show()
   
