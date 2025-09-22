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
    y:np.array = np.piecewise(x,[np.logical_and(x>-(a1+a2),x<=-a2), \
                 np.logical_and(x>-a2,x<=0), \
                 np.logical_and(x>0,x<=a3), \
                 np.logical_and(x>a3,x<=a3+a4)], \
           [lambda x: c01+x*(c11+x*(c21+x*c31)), \
            lambda x: c02+x*(c12+x*(c22+x*c32)),  \
            lambda x: c03+x*(c13+x*(c23+x*c33)),  \
            lambda x: c04+x*(c14+x*(c24+x*c34))])   
    return y 

#=========
#  MAIN  
#=========
if __name__ == "__main__":
   a1 = 1.0
   a2 = 1.0
   a3 = 0.5
   a4 = 1.5
   t = np.arange(-3.0,3.0,0.01)
   p = z1(t,a1,a2,a3,a4)
   plt.plot(t,p,linewidth=3,color="purple")
   plt.grid()
   plt.show()

