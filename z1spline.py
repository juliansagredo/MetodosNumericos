#===========================================
#  Z-spline cúbico para puntos arbitrarios
#===========================================
#  Julián T. Sagredo
#  Septiembre 2025
#  ESFM IPN
#===========================================

#=====================
#  numpy y matplotlib
#=====================
import numpy as np
import matplotlib.pyplot as plt

#=============
#  Z1-spline 
#=============
def z1(x:np.array,a1:np.float64,a2:np.float64,a3:np.float64,a4:np.float64):
    
    #==================================
    #  Coeficientes de los polinomios
    #==================================
    c01:np.float64 = (a2+a1)/a1
    c11:np.float64 = (3.0*a2+a1)/(a2*a1)
    c21:np.float64 = (3.0*a2+2.0*a1)/(a2*a1*(a2+a1))
    c31:np.float64 = 1.0/(a2*a1*(a2+a1))

    c02:np.float64 = 1.0
    c12:np.float64 = (1.0/a2)-(1.0/a3) 
    c22:np.float64 = -(a3+2.0*(a2+a1))/(a3*a2*(a2+a1))
    c32:np.float64 = -(a3+a2+a1)/(a3*a2*a2*(a2+a1))

    c03:np.float64 = 1.0
    c13:np.float64 = (1.0/a2)-(1.0/a3)
    c23:np.float64 = -(a2+2.0*(a3+a4))/(a3*a2*(a3+a4))
    c33:np.float64 = (a4+a3+a2)/(a2*a3*a3*(a3+a4))

    c04:np.float64 = (a3+a4)/a4
    c14:np.float64 = -(3.0*a3+a4)/(a3*a4)
    c24:np.float64 = (3.0*a3+2.0*a4)/(a3*a4*(a3+a4))
    c34:np.float64 = -1.0/(a3*a4*(a3+a4))
 
    #==============================
    #  Polinomio C1 por intervalos 
    #==============================
    y:np.array = np.piecewise(x,[np.logical_and(x>-(a1+a2),x<=-a2), 
                 np.logical_and(x>-a2,x<=0), 
                 np.logical_and(x>0,x<=a3), 
                 np.logical_and(x>a3,x<=a3+a4)], \
           [lambda x: c01+x*(c11+x*(c21+x*c31)), 
            lambda x: c02+x*(c12+x*(c22+x*c32)),  
            lambda x: c03+x*(c13+x*(c23+x*c33)),  
            lambda x: c04+x*(c14+x*(c24+x*c34))])   
    return y 

def z1v(x:np.array,dx1:np.float64,dx2:np.float64,dx3:np.float64,p):

    #==================================
    #  Coeficientes de los polinomios
    #==================================
    c01:np.float64 = (dx3+dx2)/dx2
    c11:np.float64 = (3.0*dx3+dx2)/(dx3*dx2)
    c21:np.float64 = (3.0*dx3+2.0*dx2)/(dx3*dx2*(dx3+dx2))
    c31:np.float64 = 1.0/(dx3*dx2*(dx3+dx2))

    c02:np.float64 = 1.0
    c12:np.float64 = (1.0/dx2)-(1.0/dx3)
    c22:np.float64 = -(dx3+2.0*(dx2+dx1))/(dx3*dx2*(dx2+dx1))
    c32:np.float64 = -(dx3+dx2+dx1)/(dx3*dx2*dx2*(dx2+dx1))

    c03:np.float64 = 1.0
    c13:np.float64 = (1.0/dx1)-(1.0/dx2)
    c23:np.float64 = -(dx1+2.0*(dx2+dx3))/(dx2*dx1*(dx2+dx3))
    c33:np.float64 = (dx3+dx2+dx1)/(dx1*dx2*dx2*(dx2+dx3))

    c04:np.float64 = (dx1+dx2)/dx2
    c14:np.float64 = -(3.0*dx1+dx2)/(dx1*dx2)
    c24:np.float64 = (3.0*dx1+2.0*dx2)/(dx1*dx2*(dx1+dx2))
    c34:np.float64 = -1.0/(dx1*dx2*(dx1+dx2))

    c05:np.float64 = 1.0 
    c15:np.float64 = -1.0/dx1
    c25:np.float64 = -1.0/(dx1*(dx1+dx2))
    c35:np.float64 = 1.0/(dx1*dx1*(dx1+dx2))

    c06:np.float64 = 0.0
    c16:np.float64 = 1.0/dx1
    c26:np.float64 = 1.0/(dx1*dx2)
    c36:np.float64 = -1.0/(dx1*dx1*dx2)

    c07:np.float64 = 0.0
    c17:np.float64 = 0.0 
    c27:np.float64 = -1.0/(dx2*(dx1+dx2))
    c37:np.float64 = 1.0/(dx1*dx2*(dx1+dx2))


    #==============================
    #  Polinomio C1 por intervalos 
    #==============================
    y:np.array = np.piecewise(x,[p == 1, p == 2, p == 3, 
                                 p == 4, p == 5, p == 6, 
                                 p == 7], \
           [lambda x: c01+x*(c11+x*(c21+x*c31)),  
            lambda x: c02+x*(c12+x*(c22+x*c32)), 
            lambda x: c03+x*(c13+x*(c23+x*c33)),  
            lambda x: c04+x*(c14+x*(c24+x*c34)),  
            lambda x: c05+x*(c15+x*(c25+x*c35)),  
            lambda x: c06+x*(c16+x*(c26+x*c36)),  
            lambda x: c07+x*(c17+x*(c27+x*c37))])
    return y


#===============================================
#  Interpolación de la función f1, dada en x1
#===============================================
def zspline(f1,x1,x2):
    n = len(x2)
    N = len(x1)
    f2 = np.zeros((n,),dtype=np.float64)
    for i in range(n):
      #================================
      # Búsqueda de índice en la malla
      #================================
      ii = int((x2[i]-x1[0])/(x1[1]-x1[0]))
      if ii>N-2: 
         ii=N-2
      while (x2[i]<x1[ii]):
         if ii==0: break
         ii = ii-1
         if ii==0: break
      while (x2[i]>x1[ii+1]):
         if ii==N-2: break
         ii = ii+1
         if ii==N-2: break

      if ii>0 and ii<N-2:
        #======================================
        #  Distancias entre puntos de la malla 
        #======================================
        dx1 = x1[ii]-x1[ii-1]
        dx2 = x1[ii+1]-x1[ii]
        dx3 = x1[ii+2]-x1[ii+1]
    
        #=======================
        #  Z1 para cada vecino 
        #=======================
        zvim1 = z1v(x2[i]-x1[ii-1],dx1,dx2,dx3,4)
        zvi   = z1v(x2[i]-x1[ii],dx1,dx2,dx3,3)
        zvip1 = z1v(x2[i]-x1[ii+1],dx1,dx2,dx3,2)
        zvip2 = z1v(x2[i]-x1[ii+2],dx1,dx2,dx3,1)

        #============================
        #  Fórmula de interpolación
        #============================
        f2[i] = f1[ii-1]*zvim1 + f1[ii]*zvi  +  f1[ii+1]*zvip1 + f1[ii+2]*zvip2

      if ii==0:
        #======================================
        #  Distancias entre puntos de la malla 
        #======================================
        dx1 = x1[1]-x1[0]
        dx2 = x1[2]-x1[1]
        dx3 = x1[3]-x1[2]

        #=======================
        #  Z1 para cada vecino 
        #=======================
        zvi   = z1v(x2[i]-x1[0],dx1,dx2,dx3,5)
        zvip1 = z1v(x2[i]-x1[0],dx1,dx2,dx3,6)
        zvip2 = z1v(x2[i]-x1[0],dx1,dx2,dx3,7)

        #============================
        #  Fórmula de interpolación
        #============================
        f2[i] = f1[0]*zvi  +  f1[1]*zvip1  + f1[2]*zvip2
 
      if ii>=N-2:
        ii = N-2
        #======================================
        #  Distancias entre puntos de la malla 
        #======================================
        dx1 = x1[N-1]-x1[N-2]
        dx2 = x1[N-2]-x1[N-3]
        dx3 = x1[N-3]-x1[N-4]

        #=======================
        #  Z1 para cada vecino 
        #=======================
        zvi   = z1v(-x2[i]+x1[N-1],dx1,dx2,dx3,5)
        zvim1 = z1v(-x2[i]+x1[N-1],dx1,dx2,dx3,6)
        zvim2 = z1v(-x2[i]+x1[N-1],dx1,dx2,dx3,7)

        #============================
        #  Fórmula de interpolación
        #============================
        f2[i] = f1[N-1]*zvi  +  f1[N-2]*zvim1  + f1[N-3]*zvim2

    return f2

#=========
#  MAIN  
#=========
if __name__ == "__main__":
   ejemplo = 2
   #================================
   # Ejemplo 1: el Z1 para ai dado 
   #================================
   if ejemplo == 1:
     a1 = 1.0
     a2 = 0.8
     a3 = 0.6
     a4 = 0.4
     t = np.arange(-2.2,2.2,0.01)
     p = z1(t,a1,a2,a3,a4)
     plt.plot(t,p,linewidth=3,color="purple")
     plt.grid()
     plt.show()

   #===========================================
   #  Ejemplo 2: interpolación de una función 
   #===========================================
   if ejemplo == 2:
     # Número de puntos
     N1 = 8 
     # Datos en posiciones al azar 
     rng = np.random.default_rng()
     x1 = rng.uniform(low=0.0, high=1.0, size=N1)
     x1 = np.sort(x1)
     #x1 = np.arange(0.0,1.0,float(1.0/N1))
     #x1 = np.append(x1,1.0)
     xe = np.arange(0.0,1.0,float(1.0/1000))
     fe = np.sin(2.0*np.pi*xe)
     x1[0] = 0.0; x1[N1-1] = 1.0
     f1 = np.sin(2.0*np.pi*x1)
     # Datos interpolados
     N2 = 200
     x2 = np.linspace(0.0,1.0,N2)
     f2 = zspline(f1,x1,x2) 
     plt.plot(xe,fe,linewidth=1,color="black")
     plt.plot(x2,f2,linewidth=4) 
     plt.plot(x1,f1,"ko")
     plt.show()
   
