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

def z1grad(f,x):
  n = len(x) 
  df = np.zeros(n,dtype=np.float64)
  for i in range(1,n-1):
    dx2:np.float64 = x[i+1]-x[i]
    dx1:np.float64 = x[i]-x[i-1]
    ci:np.float64   = 1.0/dx1 - 1.0/dx2
    cim1:np.float64 = 1.0/(dx1+dx2) - 1.0/dx1
    cip1:np.float64 = 1.0/dx2 - 1.0/(dx1+dx2)
    df[i] = (cip1*f[i+1])+(ci*f[i])+(cim1*f[i-1])
  dx10:np.float64 = x[1]-x[0]
  #dx20:np.float64 = x[2]-x[1]
  #c0:np.float64 = -(2.0*dx10+dx20)/(dx10*(dx10+dx20))
  #c1:np.float64 = 1.0/dx10 + 1.0/dx20
  #c2:np.float64 = -dx10/(dx20*(dx20+dx10))
  #df[0] = c0*f[0] + c1*f[1] + c2*f[2]
  df[0] = (f[1]-f[0])/dx10
  dx1n:np.float64 = x[n-1]-x[n-2] 
  #dx2n:np.float64 = x[n-2]-x[n-3] 
  #c0n:np.float64 = -(2.0*dx1n+dx2n)/(dx1n*(dx1n+dx2n))
  #c1n:np.float64 = 1.0/dx1n + 1.0/dx2n
  #c2n:np.float64 = -dx1n/(dx2n*(dx2n+dx1n))
  #df[n-1] = -(c0n*f[n-1]+c1n*f[n-2]+c2n*f[n-3])
  df[n-1] = (f[n-1]-f[n-2])/dx1n
  return df

#=========
#  MAIN  
#=========
if __name__ == "__main__":
   #================================
   # Ejemplo 1: el Z1 para ai dado 
   #================================
   N = 10
   x1 = np.linspace(0.0,1.0,2*N)
   f1 = np.sin(2.0*np.pi*x1)
   rng = np.random.default_rng() 
   x2 = rng.uniform(low=0.0, high=1.0, size=N)
   x2 = np.sort(x2)
   x2[0] = 0.0
   x2[N-1] = 1.0
   #x2 = x1
   f2 = np.sin(2.0*np.pi*x2) 
   df1 = np.gradient(f1,x1) 
   df2 = np.gradient(f2,x2)
   df3 = z1grad(f2,x2)
   plt.plot(x2,df2,linewidth=6,color="purple")
   plt.plot(x1,df1,linewidth=4,color="orange")
   plt.plot(x2,df3,linewidth=2,color="red")
   plt.plot(x2,df2,"o",color="black")
   plt.grid()
   plt.show()
   
