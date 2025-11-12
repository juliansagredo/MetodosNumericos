#================================================
#  Método MULTINIVEL GSGML Poisson Solver 
#================================================
#  Requiere GPU Nvidia, CUDA y NUMBA
#================================================
#  Autor: Julián T. Becerra Sagredo
#  Enero 2023
#================================================

#===========
#  Módulos
#===========
import math
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time
from numba import jit
from numba import cuda
from numba import *
from numba.types import float64,int32
from pyevtk.hl import gridToVTK

#==================
#  Función power
#==================
def power(a:int,b:int)->int:
 return a**b

#======================
#  Relajación CUDA
#======================
@cuda.jit
def relax_kernel (bc,ll,ni,nj,dt,dx,dy,err_data,t_data,rhsr_data, \
    res_data,t_array,err_array,resflag):

   i,j = cuda.grid(2)

   if i>ni-1 or j>nj-1:
      return

   if i%ll==0 and j%ll==0:

       i2d = i + j*ni
       i2d1 = i2d + ll
       i2d2 = i2d - ll
       i2d3 = i2d + ll*ni
       i2d4 = i2d - ll*ni
       i2d5 = i2d + ll + ll*ni
       i2d6 = i2d  - ll + ll*ni
       i2d7 = i2d  + ll - ll*ni
       i2d8 = i2d  - ll - ll*ni

       if bc == 0:
        if (i == 0):
         i2d2 = i2d
         i2d6 = i2d
         i2d8 = i2d
        if (j == 0): 
         i2d4 = i2d
         i2d7 = i2d
         i2d8 = i2d
        if (i == ni-1):
         i2d1 = i2d
         i2d5 = i2d
         i2d7 = i2d
        if (j == nj-1):
         i2d3 = i2d
         i2d5 = i2d
         i2d6 = i2d

       if bc == 1:
        if (i == 0):
         i2d2 = i2d + ll
         i2d6 = i2d + ll + ll*ni;
         i2d8 = i2d + ll - ll*ni;
        if (j == 0):
         i2d4 = i2d 
         i2d7 = i2d
         i2d8 = i2d
        if (i == ni-1):
         i2d1 = i2d
         i2d5 = i2d
         i2d7 = i2d
        if (j == nj-1):
         i2d3 = i2d
         i2d5 = i2d
         i2d6 = i2d


       dxl = float(ll)*dx
       dyl = float(ll)*dy
       dtl = dt*float(ll*ll)

       told = t_array[i2d]
       tip1 = t_array[i2d1]
       tim1 = t_array[i2d2]
       tjp1 = t_array[i2d3]
       tjm1 = t_array[i2d4]
       tip1jp1 = t_array[i2d5]
       tim1jp1 = t_array[i2d6]
       tip1jm1 = t_array[i2d7]
       tim1jm1 = t_array[i2d8]

       source = -rhsr_data[i2d]

       residual = ((2.0*tip1jp1+8.0*tjp1+2.0*tim1jp1+8.0*tip1 \
                  -40.0*told+8.0*tim1+2.0*tip1jm1+8.0*tjm1    \
                   +2.0*tim1jm1)/(dxl*dxl))/12.0-source

       if i==0 or j==0 or i==ni-1 or j==nj-1:
           enow = 0.0 
           residual = 0.0
       else:
           enow = dtl*residual
       tnow = told + enow
       err_data[i2d] = enow
       res_data[i2d] = residual
       if (resflag == 0):
          t_data[i2d] = tnow

#==============================
#  Interpolación lineal CUDA
#==============================
@cuda.jit
def interpolate_kernel (ll,ni,nj,dt,dx,dy,err_data,t_data, \
                rhsr_data,res_data,t_array,err_array,resflag):
    
   i,j = cuda.grid(2)
  
   if i>ni-1 or j>nj-1:
      return

   if i%ll!=0 or j%ll!=0:

      i2d = i + j*ni

      i0 = ll*int(float(i)/float(ll))
      j0 = ll*int(float(j)/float(ll))

      i0ll = i0+ll; j0ll = j0+ll;

      dxl = float(ll)*dx
      dyl = float(ll)*dy
        
      told = t_array[i2d]
      eij0 = err_array[i0+j0*ni]
      eip1 = err_array[i0ll+j0*ni]
      ejp1 = err_array[i0+j0ll*ni]
      eijp = err_array[i0ll+j0ll*ni]

      a1 = (float(i-i0)*dx)/dxl
      a0 = 1.0-a1
      b1 = (float(j-j0)*dy)/dyl
      b0 = 1.0-b1

      enow = a0*b0*eij0 + a1*b0*eip1 + a0*b1*ejp1 + a1*b1*eijp
        
      tnow = told + enow
      t_data[i2d] = tnow
        

def relax(bc,ll,dx,dy,dt,ni,nj,t_data,err_data,t_array,err_array, \
          rhsr_data,res_data,resflag):  
    
    t_array = cuda.to_device(t_data) 
    err_array = cuda.to_device(err_data)
    
    grid = (int(float(ni)/float(TILE_I))+1, \
            int(float(nj)/float(TILE_J))+1)
    block = (TILE_I, TILE_J)   
    
    relax_kernel[grid, block](bc,ll,ni,nj,dt,dx,dy,err_data, \
              t_data,rhsr_data,res_data,t_array,err_array,resflag)
    
    t_array = cuda.to_device(t_data) 
    err_array = cuda.to_device(err_data)

    interpolate_kernel[grid, block](ll,ni,nj,dt,dx,dy,err_data, \
              t_data,rhsr_data,res_data,t_array,err_array,resflag)


#=====================
#  Restricción CUDA
#=====================
@cuda.jit
def restrict_kernel(ll,ni,nj,rhs_array,rhsr_data):

   i,j = cuda.grid(2)

   if i>ni-1 or j>nj-1:
      return

   i2d = i + j*ni
	
   imll:int = i-ll
   jmll:int = j-ll
   ipll:int = i+ll
   jpll:int = j+ll
	
   if i-ll < 0:
      imll = 0
   if j-ll < 0:
      jmll = 0
   if i+ll > ni-1:
      ipll = ni-1
   if j+ll > nj-1:
      jpll = nj-1
	
   i2d1 = (imll)+ni*(jmll)
   i2d2 = (imll)+ni*(j)
   i2d3 = (imll)+ni*(jpll)
   i2d4 = (i)   +ni*(jmll)
   i2d5 = (i)   +ni*(jpll)
   i2d6 = (ipll)+ni*(jmll)
   i2d7 = (ipll)+ni*(j)
   i2d8 = (ipll)+ni*(jpll)
		  
   rij     = rhs_array[i2d]
   rim1jm1 = rhs_array[i2d1]
   rim1j   = rhs_array[i2d2]
   rim1jp1 = rhs_array[i2d3]
   rijm1   = rhs_array[i2d4]
   rijp1   = rhs_array[i2d5]
   rip1jm1 = rhs_array[i2d6]
   rip1j   = rhs_array[i2d7]
   rip1jp1 = rhs_array[i2d8]

   rtot = rij + 0.25*(rim1jm1+rim1jp1+rip1jm1+rip1jp1) \
               + 0.5*(rim1j+rijm1+rijp1+rip1j)
   rhsr_data[i2d] = rtot/4.0


def restrict(ll,ni,nj,rhs_array,rhsr_data):
    rhs_array = cuda.to_device(rhsr_data) 									   
    grid = (int(float(ni)/float(TILE_I))+1, \
            int(float(nj)/float(TILE_J))+1)
    block = (TILE_I, TILE_J)
    restrict_kernel[grid, block](ll,ni,nj,rhs_array,rhsr_data)

 	
#============================
#  Suma de soluciones CUDA
#============================
@cuda.jit
def suma_kernel(ni,nj,t_data,tt_data):
   i,j = cuda.grid(2)
   if i>ni-1 or j>nj-1:
      return
   i2d = i + j*ni
   tt_data[i2d] += t_data[i2d]
   t_data[i2d] = 0.0

def suma(ni,nj,t_data,tt_data):
    grid = (int(float(ni)/float(TILE_I))+1, \
            int(float(nj)/float(TILE_J))+1)
    block = (TILE_I, TILE_J)
    suma_kernel[grid, block](ni,nj,t_data,tt_data)

#=================================
#  Condiciones de frontera CUDA
#=================================
@cuda.jit
def apply_BCs_kernel(ni,nj,t_data,err_data):
   i,j = cuda.grid(2) 
   if i>ni-1 or j>nj-1:
      return
   i2d = i + j*ni
   if i == 0 or i==ni-1 or j == 0 or j==nj-1:
      t_data[i2d] = 0.0
   if i == 0 or i==ni-1 or j == 0 or j==nj-1:
      err_data[i2d] = 0.0

def apply_BCs(ni,nj,t_data,err_data): 
    grid = (int(float(ni)/float(TILE_I))+1, \
            int(float(nj)/float(TILE_J))+1)
    block = (TILE_I, TILE_J)
    apply_BCs_kernel[grid, block](ni, nj, t_data, err_data)

#==========================
#  Frontera inmersa CUDA 
#==========================
@cuda.jit
def inmersa_kernel(l,ni,nj,dx,dy,ii_data,ic_data,xc_data,yc_data, \
                   t_array,rhs_data,ju_data,jun_data):

   i,j = cuda.grid(2)

   if i>ni-1 or j>nj-1: 
      return

   # Indice del punto en la malla
   i2d = i+j*ni
   ii = ii_data[i2d]

   #-----------------------------------------------------
   #  Para todos los puntos cercanos a la curva inmersa
   #-----------------------------------------------------
   if ii != 0:

     # Signo del punto central
     signo = ii

     #----------------------------
     #  Indices de los 9 vecinos
     #----------------------------
     i2d1 = (i+1) + j*ni
     i2d2 = (i-1) + j*ni
     i2d3 = i + (j+1)*ni
     i2d4 = i + (j-1)*ni
     i2d5 = (i+1) + (j+1)*ni
     i2d6 = (i-1) + (j+1)*ni
     i2d7 = (i+1) + (j-1)*ni
     i2d8 = (i-1) + (j-1)*ni

     #--------------------
     #  Datos de vecinos
     #--------------------
     tt = cuda.local.array(9,dtype=float64)
     tt[0] = t_array[i2d]
     tt[1] = t_array[i2d1]
     tt[2] = t_array[i2d2]
     tt[3] = t_array[i2d3]
     tt[4] = t_array[i2d4]
     tt[5] = t_array[i2d5]
     tt[6] = t_array[i2d6]
     tt[7] = t_array[i2d7]
     tt[8] = t_array[i2d8]
 
     #--------------------
     #  Vectores radiales
     #--------------------
     drx = cuda.local.array(9,dtype=float64)
     dry = cuda.local.array(9,dtype=float64)
     drx[0] = 0.0;   dry[0] = 0.0;
     drx[1] =  dx;   dry[1] = 0.0;
     drx[2] = -dx;   dry[2] = 0.0;
     drx[3] = 0.0;   dry[3] = dy;
     drx[4] = 0.0;   dry[4] = -dy;
     drx[5] =  dx;   dry[5] = dy;
     drx[6] = -dx;   dry[6] = dy;
     drx[7] =  dx;   dry[7] = -dy;
     drx[8] = -dx;   dry[8] = -dy;

     #-------------------------------------------
     #  Signo de los puntos cercanos a la curva
     #-------------------------------------------
     sv = cuda.local.array(9,dtype=int32)
     sv[0] = ii_data[i2d]
     sv[1] = ii_data[i2d1]
     sv[2] = ii_data[i2d2]
     sv[3] = ii_data[i2d3]
     sv[4] = ii_data[i2d4]
     sv[5] = ii_data[i2d5]
     sv[6] = ii_data[i2d6]
     sv[7] = ii_data[i2d7]
     sv[8] = ii_data[i2d8]

     #------------------------------------------------------
     #  ss es 0 si está del mismo lado que el nodo central,
     #  1 si el nodo es +1 o -1 si el nodo es -1
     #------------------------------------------------------
     ss = cuda.local.array(9,dtype=float64)
     ss[0] = 0.5*float(signo-sv[0])*float(abs(sv[0]))
     ss[1] = 0.5*float(signo-sv[1])*float(abs(sv[1]))
     ss[2] = 0.5*float(signo-sv[2])*float(abs(sv[2]))
     ss[3] = 0.5*float(signo-sv[3])*float(abs(sv[3]))
     ss[4] = 0.5*float(signo-sv[4])*float(abs(sv[4]))
     ss[5] = 0.5*float(signo-sv[5])*float(abs(sv[5]))
     ss[6] = 0.5*float(signo-sv[6])*float(abs(sv[6]))
     ss[7] = 0.5*float(signo-sv[7])*float(abs(sv[7]))
     ss[8] = 0.5*float(signo-sv[8])*float(abs(sv[8]))
  
     sss = cuda.local.array(9,dtype=float64)
     for  k in range(9):
        if signo==sv[k]:
           sss[k] = -signo

     #-------------------------
     #  Datos sobre la curva 
     #-------------------------
     ic = ic_data[i2d]

     """
     # Cuidado con primer y último vecino
     icp1 = ic+1
     icm1 = ic-1
     if ic == 0:
        icm1 = numtt-1
     if ic == numtt-1:
        icp1 = 0
     xicp1 = xc_data[icp1]
     yicp1 = yc_data[icp1]
     xicm1 = xc_data[icm1]
     yicm1 = yc_data[icm1]
     xs = xicp1 - xicm1
     ys = yicp1 - yicm1
     oor = 1.0/math.sqrt(xs*xs+ys*ys)
     if oor == 0.0: 
        oor = 1.0
     """
     alpha1 = xc_data[ic]
     alpha2 = yc_data[ic]

     #iin = icn_data[i2d]

     #----------------------------------
     #  Añadir la fuente de los saltos
     #----------------------------------
     if l==0:
       source = 0.0
       ju = ju_data[ii]
       #jun = jun_data[iin]
       #jux = oor*ys*jun
       #juy = oor*xs*jun
       for k in range(1,9):
         if ss[k] != 0: 
           if k==1:
              iii=i+1 
              jjj=j 
              ohlpq=(8.0/(dx*dx))/12.0
           if k==2: 
              iii=i-1 
              jjj=j 
              ohlpq=(8.0/(dx*dx))/12.0
           if k==3: 
              iii=i 
              jjj=j+1 
              ohlpq=(8.0/(dx*dx))/12.0
           if k==4: 
              iii=i 
              jjj=j-1 
              ohlpq=(8.0/(dx*dx))/12.0
           if k==5: 
              iii=i+1 
              jjj=j+1 
              ohlpq=(2.0/(dx*dx))/12.0
           if k==6: 
              iii=i-1 
              jjj=j+1 
              ohlpq=(2.0/(dx*dx))/12.0
           if k==7: 
              iii=i+1 
              jjj=j-1 
              ohlpq=(2.0/(dx*dx))/12.0
           if k==8: 
              iii=i-1 
              jjj=j-1 
              ohlpq=(2.0/(dx*dx))/12.0
         xx = iii*dx  # coordenada x vecino en la malla
         yy = jjj*dy  # coordenada y vecino en la malla
         #source += ss[k]*ohlpq*(ju+jux*(xx-alpha1)+juy*(yy-alpha2))
         source += ss[k]*ohlpq*ju

       rhs_data[i2d] -= source


     """
     // Calcular el salto a partir del gradiente numerico
     if (l==1) {
         Real vnx = -ys*oor;
         Real vny = xs*oor;
         Real gt1x = 0.f;
         Real gt1y = 0.f;
	 Real gt2x = 0.f;
	 Real gt2y = 0.f;
         Real ju = ju_data[iin];
         Real jun = jun_data[iin];
         int iii, jjj, ttt, tt0;
	 Real jux = oor*ys*jun;
         Real juy = oor*xs*jun;
	 Real alpha1 = xc_data[ic];
         Real alpha2 = yc_data[ic];

         for (int k=1; k<9; k++) {

	   if (k==1) {iii=i+1; jjj=j;}
           if (k==2) {iii=i-1; jjj=j;}
           if (k==3) {iii=i; jjj=j+1;}
           if (k==4) {iii=i; jjj=j-1;}
           if (k==5) {iii=i+1; jjj=j+1;}
           if (k==6) {iii=i-1; jjj=j+1;}
           if (k==7) {iii=i+1; jjj=j-1;}
           if (k==8) {iii=i-1; jjj=j-1;}

	   Real xx = iii*dx;
           Real yy = jjj*dy;
	   if (ss[k] != 0) {
             ttt = tt[k] + ss[k]*(ju+jux*(xx-alpha1)+juy*(yy-alpha2));
             gt1x += 0.25f*(ttt-tt[0])*drx[k]/(drx[k]*drx[k]+dry[k]*dry[k]);
             gt1y += 0.25f*(ttt-tt[0])*dry[k]/(drx[k]*drx[k]+dry[k]*dry[k]);
	   } else {
             ttt = tt[k]+sss[k]*(ju+jux*(xx-alpha1)+juy*(yy-alpha2));
	     tt0 = tt[0]+sss[0]*(ju+jux*(i*dx-alpha1)+juy*(j*dy-alpha2));
	     gt2x += 0.25f*(tt[k]-tt[0])*drx[k]/(drx[k]*drx[k]+dry[k]*dry[k]);
             gt2y += 0.25f*(tt[k]-tt[0])*dry[k]/(drx[k]*drx[k]+dry[k]*dry[k]);
           }

         }

         Real gt1n = gt1x*vnx + gt1y*vny;
         Real gt2n = gt2x*vnx + gt2y*vny;
         Real jf_data = 0.f;

       }

     }
   }
}

"""
def inmersa(l,ni,nj,dx,dy,ii_data,ic_data,xc_data,yc_data,t_array,\
            rhs_data,ju_data,jun_data):

    grid = (int(float(ni)/float(TILE_I))+1,                       \
            int(float(nj)/float(TILE_J))+1)
    block = (TILE_I, TILE_J)

    inmersa_kernel[grid, block](l,ni,nj,dx,dy,ii_data,ic_data,    \
            xc_data,yc_data,t_array,rhs_data,ju_data,jun_data)

    
#===============
#  Multilevel
#===============
def Multilevel(bc,ni,nj,res_data,err_host,rhs_data,rhsr_data,dx,  \
               dy,dt,t_data,err_data,t_array,err_array,rhs_array, \
               rhs_host,tt_data,zeros,ii_data,ic_data,xc_data,    \
               yc_data,ju_data,jun_data):

  print("Entrando a Poisson solver GSGML...")

  rhs_data = cuda.to_device(rhs_host) 

  ntot = ni*nj

  #-----------------------
  # parametros del ciclo
  #-----------------------
  npot = int(np.log2(float((ni-1)/2)))
  nrel = ni-1
  nnrel = 2 
  nvci = npot
  nmin = 2 
  cycles = 12

  print("ciclos = ",cycles)

  #------------------------------
  # Iteraciones de los residuos
  #------------------------------
  for ncyc in range(cycles):

   print("ciclo = ",ncyc)
   ndown = npot-2

   if ncyc>0: 
    # residual 
    #rhs_data = cuda.to_device(res_data) 
    res_data.copy_to_host(rhs_host)
    err_data   = cuda.to_device(zeros) 
    
   #---------------------
   #  Ciclo de cerrucho  
   #---------------------
   for nni in range(nvci):
 
    #--------------------------
    #  De lo grueso a lo fino
    #--------------------------
    for nn2 in range(ndown+2,ndown,-1):
     nnl = power(2,nn2)
     nrelax = int(nrel/nnl)
     nrelax = min(nrelax,nmin)
     rhsr_data = cuda.to_device(rhs_host)
     if ncyc==0:
        inmersa(0,ni,nj,dx,dy,ii_data,ic_data,xc_data,yc_data, \
                t_array,rhsr_data,ju_data,jun_data)
     for nn3 in range(nn2):
      nn4 = power(2,nn3)
      restrict(nn4,ni,nj,rhs_array,rhsr_data)
     for nn in range(nrelax):
      relax(bc,nnl,dx,dy,dt,ni,nj,t_data,err_data,t_array, \
            err_array,rhsr_data,res_data,0)
      #apply_BCs(ni, nj, t_data, err_data)
     
    #--------------------------
    #  De lo fino a lo grueso
    #--------------------------
    for nn2 in range(ndown+2,ndown+3):
     nnl = power(2,nn2)
     nrelax = int(nrel/nnl)
     nrelax = min(nrelax,nmin)
     rhsr_data = cuda.to_device(rhs_host)
     if ncyc==0:
        inmersa(0,ni,nj,dx,dy,ii_data,ic_data,xc_data,yc_data, \
                t_array,rhsr_data,ju_data,jun_data)
     for nn3 in range(nn2):
      nn4 = power(2,nn3)
      restrict(nn4,ni,nj,rhs_array,rhsr_data)
     for nn in range(nrelax):
      relax(bc,nnl,dx,dy,dt,ni,nj,t_data,err_data,t_array, \
            err_array,rhsr_data,res_data,0)
      #apply_BCs(ni, nj, t_data, err_data)
     
    ndown -= 1
    
   #----------------- 
   #  A lo más fino 
   #-----------------
   for nni in range(nnrel):
    for nn2 in range(ndown+2,-1,-1):
     nnl = power(2,nn2)
     nrelax = int(nrel/nnl)
     nrelax = min(nrelax,nmin)
     rhsr_data = cuda.to_device(rhs_host)
     if ncyc==0:
        inmersa(0,ni,nj,dx,dy,ii_data,ic_data,xc_data,yc_data, \
                t_array,rhsr_data,ju_data,jun_data)
     for nn3 in range(nn2):
      nn4 = power(2,nn3)
      restrict(nn4,ni,nj,rhs_array,rhsr_data)
      count +=1
     for nn in range(nrelax):
      relax(bc,nnl,dx,dy,dt,ni,nj,t_data,err_data,t_array, \
            err_array,rhsr_data,res_data,0)
      #apply_BCs(ni, nj, t_data, err_data)
     
   #-------------------
   #  Último residuo
   #-------------------
   rhsr_data = cuda.to_device(rhs_host)
   for nn in range(1): 
     if ncyc==0:
        inmersa(0,ni,nj,dx,dy,ii_data,ic_data,xc_data,yc_data, \
                t_array,rhsr_data,ju_data,jun_data)
     relax(bc,1,dx,dy,dt,ni,nj,t_data,err_data,t_array,err_array, \
           rhsr_data,res_data,1)
   #apply_BCs(ni, nj, t_data, err_data)

   #----------------------
   # Sumar a la solución
   #----------------------
   suma(ni,nj,t_data,tt_data)


#=======================================
#  Lectura y parametrización de curvas
#=======================================
def leer_curva(curvas,dx):

   #-------------------------------------------
   #  Lectura y escala de puntos en las curvas
   #-------------------------------------------
   npuntos = np.zeros(curvas,dtype=np.int32)
   scalecx = np.zeros(curvas,dtype=np.float64)
   scalecy = np.zeros(curvas,dtype=np.float64)
   xc0 = np.zeros(curvas,dtype=np.float64)
   yc0 = np.zeros(curvas,dtype=np.float64)
  
   #--------------------
   #  Leer la dimensión
   #--------------------
   tp = 0
   for ic in range(curvas):
      with open("./PUNTOS/curva"+str(ic)+".txt") as f:
         npuntos[ic], scalecx[ic], scalecy[ic], xc0[ic], \
             yc0[ic] = [float(x) for x in next(f).split()]
      print ("npuntos curva "+str(ic+1) +" = ", npuntos[ic])
      print ("xc0 y scalecx = ",xc0[ic],scalecx[ic])
      print ("yc0 y scalecy = ",yc0[ic],scalecy[ic])
      tp += npuntos[ic]
      f.close()
   print("Puntos leídos = ",tp)

   x_curva = np.zeros(tp,dtype=np.float64)
   y_curva = np.zeros(tp,dtype=np.float64)
   ll = np.zeros(tp,dtype=np.int32)
 
   #------------------
   #  Leer las curvas
   #------------------
   tp = 0
   for ic in range(curvas):
     with open("./PUNTOS/curva"+str(ic)+".txt") as f:
         npuntos[ic], scalecx[ic], scalecy[ic], xc0[ic], \
             yc0[ic] = [float(x) for x in next(f).split()]
         for i in range(npuntos[ic]):
           ll[i+tp],xc,yc  = [float(x) for x in next(f).split()] 
           x_curva[i+tp] = float(xc0[ic] + scalecx[ic]*xc) 
           y_curva[i+tp] = float(yc0[ic] + scalecy[ic]*yc)
     tp += npuntos[ic]
     f.close() 

   #---------------------------------------
   #  Longitud y número de puntos nuevos
   #---------------------------------------
   numt = np.zeros(curvas,dtype=np.int32)
   ds = np.zeros(curvas,dtype=np.float64)
   dxx= np.zeros(curvas,dtype=np.float64)
   alphadx = 0.1
   numtt = 0 
   tp = 0 
   for ic in range(curvas):
     ds[ic] = np.sqrt((x_curva[tp+0]-x_curva[tp+1])  \
                     *(x_curva[tp+0]-x_curva[tp+1])+ \
                      (y_curva[tp+0]-y_curva[tp+1])* \
                      (y_curva[tp+0]-y_curva[tp+1]))
     numt[ic] = int(float(npuntos[ic])*ds[ic]/(alphadx*dx[0]))
     dxx[ic] = float(npuntos[ic])*ds[ic]/float(numt[ic])
     numtt += numt[ic]
     print("Puntos en curva "+str(ic)+" = ",numt[ic])
     tp += npuntos[ic]
   print("Total de puntos en las curvas = ",numtt)

   larc = np.zeros(numtt,dtype=np.float64)
   xx   = np.zeros(numtt,dtype=np.float64) 
   yy   = np.zeros(numtt,dtype=np.float64) 
   tt   = np.zeros(numtt,dtype=np.float64)

   l0 = 2
   tp = 0
   ttp = 0
   for ic in range(curvas): 
    #-----------------------------------------------------
    # Generar puntos sobre la curva con longitud de arco 
    #-----------------------------------------------------
    for iip in range(numt[ic]): 
      larc[ttp+iip] = float(iip)*(dxx[ic]/ds[ic])
	                                  
    for ipunt in range(numt[ic]): 
      l = l0;
      idx = int(larc[ttp+ipunt]);

      if (idx>0 and idx < npuntos[ic]-2):
        if (ll[tp+idx-1] == 0 or ll[tp+idx+2]==0): 
          l=1
      elif (idx == 0):
        if (ll[tp+npuntos[ic]-1] == 0 or ll[tp+idx+2]==0):
          l=1
      elif (idx == npuntos[ic]-2):
        if (ll[tp+npuntos[ic]-3] == 0 or ll[tp+0] == 0):
          l=1
      elif (idx == npuntos[ic]-1): 
        if (ll[tp+npuntos[ic]-2] == 0 or ll[tp+1] == 0):
          l=1
      elif (idx == npuntos[ic]):
        if (ll[tp+npuntos[ic]-1] == 0 or ll[tp+2] == 0):
          l=1

      if (idx<npuntos[ic]-1): 
        if (ll[tp+idx] == 0 or ll[tp+idx+1] == 0): 
          l=0
      else:
        if (ll[tp+npuntos[ic]-1]==0 or ll[tp+0] ==0):
          l=0

      #---------------------------
      # Interpolacion quintica Z2
      #---------------------------
      if(l==2):

        xx2 = larc[ttp+ipunt]-float(idx)
        xxx0 = xx2+2.0
        xx1 = xx2+1.0
        xx3 = 1.0-xx2
        xx4 = 2.0-xx2
        xx5 = 3.0-xx2
        a0 = 18.0+xxx0*((-153.0/4.0)+xxx0*((255.0/8.0)+xxx0*((    \
             -313.0/24.0)+xxx0*((21.0/8.0)+(-5.0/24.0)*xxx0))))
        a1 = -4.0+xx1*((75.0/4.0)+xx1*((-245.0/8.0) + xx1*((545.0 \
             /24.0)+xx1*((-63.0/8.0)+xx1*(25.0/24.0)))))
        a2 = 1.0+xx2*xx2*((-15.0/12.0)+xx2*((-35.0/12.0)+xx2*((   \
             63.0/12.0)+xx2*(-25.0/12.0))))
        a3 = 1.0+xx3*xx3*((-15.0/12.0)+xx3*((-35.0/12.0)+xx3*((   \
             63.0/12.0)+xx3*(-25.0/12.0))))
        a4 = -4.0+xx4*((75.0/4.0)+xx4*((-245.0/8.0)+xx4*((545.0/  \
             24.0)+xx4*((-63.0/8.0)+xx4*(25.0/24.0)))))
        a5 = 18.0+xx5*((-153.0/4.0)+xx5*((255.0/8.0)+xx5*((       \
             -313.0/24.0)+xx5*((21.0/8.0)+(-5.0/24.0)*xx5))))

        if (idx<npuntos[ic]):
            ax2 = x_curva[tp+idx] 
        else:
            ax2 = x_curva[tp+0]

        if (idx<npuntos[ic]):
            ay2 = y_curva[tp+idx] 
        else: 
            ay2 = y_curva[tp+0]

        if (idx>1): 
          ax0 = x_curva[tp+idx-2] 
        elif (idx>0): 
          ax0=x_curva[tp+npuntos[ic]-1]
        else:
          ax0=x_curva[tp+npuntos[ic]-2]

        if (idx>0):
          ax1 = x_curva[tp+idx-1] 
        else: 
          ax1 = x_curva[tp+npuntos[ic]-1]

        if (idx<npuntos[ic]-1):
          ax3 = x_curva[tp+idx+1] 
        else: 
          ax3 = x_curva[tp+0]

        if (idx<npuntos[ic]-2):
          ax4 = x_curva[tp+idx+2] 
        elif (idx<npuntos[ic]-1): 
          ax4 = x_curva[tp+0]
        else: 
          ax4=x_curva[tp+1]

        if (idx<npuntos[ic]-3):
          ax5 = x_curva[tp+idx+3] 
        elif (idx<npuntos[ic]-2): 
          ax5 = x_curva[tp+0]
        elif (idx<npuntos[ic]-1): 
          ax5 = x_curva[tp+1] 
        else:
          ax5 = x_curva[tp+2]

        if (idx>0):
          ay1 = y_curva[tp+idx-1] 
        else: 
          ay1 = y_curva[tp+npuntos[ic]-1]

        if (idx>1):
          ay0 = y_curva[tp+idx-2] 
        elif(idx>0):  
          ay0=y_curva[tp+npuntos[ic]-1]
        else:
          ay0=y_curva[tp+npuntos[ic]-2]

        if (idx<npuntos[ic]-1):
          ay3 = y_curva[tp+idx+1] 
        else: 
          ay3 = y_curva[tp+0]

        if (idx<npuntos[ic]-2):
          ay4 = y_curva[tp+idx+2] 
        elif(idx<npuntos[ic]-1): 
          ay4 = y_curva[tp+0]
        else:
          ay4=y_curva[tp+1]

        if (idx<npuntos[ic]-3):
          ay5 = y_curva[tp+idx+3] 
        elif (idx<npuntos[ic]-2):
          ay5 = y_curva[tp+0]
        elif (idx<npuntos[ic]-1): 
          ay5 = y_curva[tp+1] 
        else:
          ay5 = y_curva[tp+2]

        xx[ttp+ipunt] =  a0*ax0+a1*ax1+a2*ax2+a3*ax3+a4*ax4+a5*ax5
        yy[ttp+ipunt] =  a0*ay0+a1*ay1+a2*ay2+a3*ay3+a4*ay4+a5*ay5

      #--------------------------
      # Interpolacion cubica Z1
      #--------------------------
      if(l==1):

        xx1 = larc[ttp+ipunt]-float(idx)
        xxx0 = xx1+1.0
        xx2 = 1.0-xx1
        xx3 = 2.0-xx1
        a0 = 0.50*(2.0-xxx0)*(2.0-xxx0)*(1.0-xxx0)
        a1 = 1.0-2.50*(xx1*xx1)+1.50*(xx1*xx1*xx1)
        a2 = 1.0-2.50*(xx2*xx2)+1.50*(xx2*xx2*xx2)
        a3 = 0.50*(2.0-xx3)*(2.0-xx3)*(1.0-xx3)

        if (idx>0):
           ax0 = x_curva[tp+idx-1] 
        else: 
           ax0 = x_curva[tp+npuntos[ic]-1]

        if (idx<npuntos[ic]):
           ax1 = x_curva[tp+idx] 
        else: 
           ax1 = x_curva[tp+0]

        if (idx<npuntos[ic]-1):
           ax2 = x_curva[tp+idx+1] 
        elif (idx<npuntos[ic]): 
           ax2 = x_curva[tp+0]
        else: 
           ax2 = x_curva[tp+1]

        if (idx<npuntos[ic]-2):
           ax3 = x_curva[tp+idx+2] 
        elif (idx<npuntos[ic]-1): 
           ax3 = x_curva[tp+0]
        elif (idx<npuntos[ic]):
           ax3 = x_curva[tp+1] 
        else: 
           ax3 = x_curva[tp+2]

        if (idx>0):
           ay0 = y_curva[tp+idx-1] 
        else: 
           ay0 = y_curva[tp+npuntos[ic]-1]

        if (idx<npuntos[ic]):
           ay1 = y_curva[tp+idx] 
        else: 
           ay1 = y_curva[tp+0]

        if (idx<npuntos[ic]-1):
           ay2 = y_curva[tp+idx+1] 
        elif (idx<npuntos[ic]): 
           ay2 = y_curva[tp+0]
        else:
           ay2 = y_curva[tp+1]

        if (idx<npuntos[ic]-2): 
           ay3 = y_curva[tp+idx+2] 
        elif (idx<npuntos[ic]-1): 
           ay3 = y_curva[tp+0]
        elif (idx<npuntos[ic]): 
           ay3 = y_curva[tp+1] 
        else:
           ay3 = y_curva[tp+2]

        xx[ttp+ipunt] =  a0*ax0+a1*ax1+a2*ax2+a3*ax3
        yy[ttp+ipunt] =  a0*ay0+a1*ay1+a2*ay2+a3*ay3

      #--------------------------
      # Interpolacion lineal Z0 
      #--------------------------
      if(l==0):
        xxx0 = larc[ttp+ipunt]-float(idx)
        xx1 = 1.0-xxx0
        a0 = xx1
        a1 = xxx0
        if (idx<npuntos[ic]):
           ax0 = x_curva[tp+idx] 
        else:
           ax0 = x_curva[tp+0]
        if (idx<npuntos[ic]-1):
           ax1 = x_curva[tp+idx+1] 
        elif (idx<npuntos[ic]): 
           ax1 = x_curva[tp+0]
        else: 
           ax1 = x_curva[tp+1]
        if (idx<npuntos[ic]):
           ay0 = y_curva[tp+idx] 
        else: 
           ay0 = y_curva[tp+0]
        if (idx<npuntos[ic]-1):
           ay1 = y_curva[tp+idx+1] 
        elif (idx < npuntos[ic]): 
           ay1 = y_curva[tp+0]
        else: 
           ay1 = y_curva[tp+1]

        xx[ttp+ipunt] =  a0*ax0+a1*ax1
        yy[ttp+ipunt] =  a0*ay0+a1*ay1

    ttp += numt[ic]
    tp  += npuntos[ic]

   return npuntos,numt,xx,yy

#========================================
#  Punto más cercano a P en segmento AB 
#========================================
def punto_cercano(x1,y1,x2,y2,x3,y3):
    AB = np.array([x2-x1,y2-y1],dtype=np.float64)  
    AP = np.array([x3-x1,y3-y1],dtype=np.float64)

    magnitud = np.sqrt(np.dot(AB,AB))
    ABdotAP  = np.dot(AB,AP) 
    distance = ABdotAP/magnitud 

    if distance < 0.0:
        return 0.0,x1,y1
    elif distance > 1.0:
        return 1.0,x2,y2
    else:
        return distance, x1 + AB[0]*distance, y1 + AB[1]*distance


#===============================================
#  Marcar malla cercana a la frontera inmersa
#===============================================
def marcar_malla(n,curvas,numt,xx,yy):
   
   #-----------------------------------------
   # Marcar los vecinos de la curva inmersa 
   #-----------------------------------------
   nt = n[0]*n[1] 
   ii_host = np.zeros(nt,dtype=np.int32)
   ic_host = np.zeros(nt,dtype=np.int32)
   dc_host = np.zeros(nt,dtype=np.float64)

   for i in range(nt):
     dc_host[i] = -1.0 

   tp = 0
   ttp = 0
   for ic in range(curvas):

    for k in range(ttp,ttp+numt[ic]):

      km1 = k-1
      kp1 = k+1

      if k == ttp: 
        km1 = ttp+numt[ic]-1
      if k == ttp+numt[ic]-1:
        kp1 = ttp;

      d2x = (xx[kp1]-xx[k])
      d2y = (yy[kp1]-yy[k])

      iii = int(xx[k]/dx[0])
      jjj = int(yy[k]/dx[1])
      inn = iii+n[0]*jjj

      inn2 = (iii+1)+n[0]*jjj
      inn3 = (iii)+n[0]*(jjj+1)
      inn4 = (iii+1)+n[0]*(jjj+1)

      #-------------------------------------------------
      # Establecer distancia a primeros puntos vecinos
      #-------------------------------------------------
      pv1x = dx[0]*float(iii) 
      pv1y = dx[1]*float(jjj)
      pv2x = dx[0]*float(iii+1)
      pv2y = dx[1]*float(jjj)
      pv3x = dx[0]*float(iii)
      pv3y = dx[1]*float(jjj+1)
      pv4x = dx[0]*float(iii+1)
      pv4y = dx[1]*float(jjj+1)

      if k == ttp:
         dc_host[inn]  = np.sqrt((pv1x-xx[k])**2+(pv1y-yy[k])**2)
         dc_host[inn2] = np.sqrt((pv2x-xx[k])**2+(pv2y-yy[k])**2)
         dc_host[inn3] = np.sqrt((pv3x-xx[k])**2+(pv3y-yy[k])**2)
         dc_host[inn4] = np.sqrt((pv4x-xx[k])**2+(pv4y-yy[k])**2)

         ic_host[inn]  = k
         ic_host[inn2] = k
         ic_host[inn3] = k
         ic_host[inn4] = k

      #------------------------------------------
      #  Establecer distancia mínima a segmento 
      #------------------------------------------
      d1,p11,p12 = punto_cercano(xx[k],yy[k],xx[kp1],yy[kp1],pv1x,pv1y)
      d2,p21,p22 = punto_cercano(xx[k],yy[k],xx[kp1],yy[kp1],pv2x,pv2y)
      d3,p31,p32 = punto_cercano(xx[k],yy[k],xx[kp1],yy[kp1],pv3x,pv3y)
      d4,p41,p42 = punto_cercano(xx[k],yy[k],xx[kp1],yy[kp1],pv4x,pv4y)

      id1 = np.rint(d1)
      id2 = np.rint(d2)
      id3 = np.rint(d3)
      id4 = np.rint(d4)

      dist1 = np.sqrt((pv1x - p11)**2 + (pv1y - p12)**2)
      dist2 = np.sqrt((pv2x - p21)**2 + (pv2y - p22)**2)
      dist3 = np.sqrt((pv3x - p31)**2 + (pv3y - p32)**2)
      dist4 = np.sqrt((pv4x - p41)**2 + (pv4y - p42)**2) 
 
      cruz1 = d2x*(pv1y - p12)-d2y*(pv1x - p11)
      cruz2 = d2x*(pv2y - p22)-d2y*(pv2x - p21)
      cruz3 = d2x*(pv3y - p32)-d2y*(pv3x - p31)
      cruz4 = d2x*(pv4y - p42)-d2y*(pv4x - p41)

      if dc_host[inn] == -1.0:
          dc_host[inn] = dist1
          ic_host[inn] = k+id1
          if cruz1 < 0:
              ii_host[inn] = -1
          else:
              ii_host[inn] = 1

      if dc_host[inn2] == -1.0:
          dc_host[inn2] = dist2
          ic_host[inn2] = k+id2
          if cruz2 < 0:
              ii_host[inn2] = -1
          else:
              ii_host[inn2] = 1

      if dc_host[inn3] == -1.0:
          dc_host[inn3] = dist3
          ic_host[inn3] = k+id3
          if cruz3 < 0:
              ii_host[inn3] = -1
          else:
              ii_host[inn3] = 1

      if dc_host[inn4] == -1.0:
          dc_host[inn4] = dist4 
          ic_host[inn4] = k+id4
          if cruz4 < 0:
              ii_host[inn4] = -1
          else:
              ii_host[inn4] = 1

      if dist1<dc_host[inn]:
          dc_host[inn] = dist1
          ic_host[inn] = k+id1
          if cruz1 < 0:
              ii_host[inn] = -1
          else:
              ii_host[inn] = 1

      if dist2<dc_host[inn2]:
          dc_host[inn2] = dist2
          ic_host[inn2] = k+id2
          if cruz2 < 0:
              ii_host[inn2] = -1
          else:
              ii_host[inn2] = 1

      if dist3<dc_host[inn3]:
          dc_host[inn3] = dist3
          ic_host[inn3] = k+id3
          if cruz3 < 0:
              ii_host[inn3] = -1
          else:
              ii_host[inn3] = 1

      if dist4<dc_host[inn4]:
          dc_host[inn4] = dist4
          ic_host[inn4] = k+id4
          if cruz4 < 0:
              ii_host[inn4] = -1
          else:
              ii_host[inn4] = 1

      #---------------------
      #  Celda intermedia 
      #---------------------
      iip1 = int(xx[kp1]/dx[0])
      jjp1 = int(yy[kp1]/dx[1]) 

      if iip1 != iii and jjp1 != jjj:

         v1 = iip1-iii
         v2 = jjp1-jjj

         # Checar esquinas
         if v1==1 and v2==1:
             if ii_host[inn4] == -1:
                ninn = iii+(jjj+2)*n[0]
                ii_host[ninn] = 1
                ppx = float(iii)*dx[0]
                ppy = float(jjj+2)*dx[1]
                dd,p1,p2 = punto_cercano(xx[k],yy[k],xx[kp1],yy[kp1],ppx,ppy)
                idd = np.rint(dd)
                dist = np.sqrt((ppx - p1)**2 + (ppy - p2)**2)
                if dc_host[ninn] == -1:
                   dc_host[ninn] = dist
                   ic_host[ninn] = k+idd
             else:
                ninn = iii+2+(jjj)*n[0]
                ii_host[ninn] = -1
                ppx = float(iii+2)*dx[0]
                ppy = float(jjj)*dx[1]
                dd,p1,p2 = punto_cercano(xx[k],yy[k],xx[kp1],yy[kp1],ppx,ppy)
                idd = np.rint(dd)
                dist = np.sqrt((ppx - p1)**2 + (ppy - p2)**2)
                if dc_host[ninn] == -1:
                   dc_host[ninn] = dist
                   ic_host[ninn] = k+idd
         if v1==-1 and v2==1:
             if ii_host[inn3] == -1:
                ninn = iii-1+(jjj)*n[0]
                ii_host[ninn] = 1
                ppx = float(iii-1)*dx[0]
                ppy = float(jjj)*dx[1]
                dd,p1,p2 = punto_cercano(xx[k],yy[k],xx[kp1],yy[kp1],ppx,ppy)
                idd = np.rint(dd)
                dist = np.sqrt((ppx - p1)**2 + (ppy - p2)**2)
                if dc_host[ninn] == -1:
                   dc_host[ninn] = dist
                   ic_host[ninn] = k+idd
             else:
                ninn = iii+1+(jjj+2)*n[0]
                ii_host[ninn] = -1
                ppx = float(iii+1)*dx[0]
                ppy = float(jjj+2)*dx[1]
                dd,p1,p2 = punto_cercano(xx[k],yy[k],xx[kp1],yy[kp1],ppx,ppy)
                idd = np.rint(dd)
                dist = np.sqrt((ppx - p1)**2 + (ppy - p2)**2)
                if dc_host[ninn] == -1:
                   dc_host[ninn] = dist
                   ic_host[ninn] = k+idd
         if v1==1 and v2==-1:
             if ii_host[inn2] == -1:
                ninn = iii+2+(jjj+1)*n[0]
                ii_host[ninn] = 1
                ppx = float(iii+2)*dx[0]
                ppy = float(jjj+1)*dx[1]
                dd,p1,p2 = punto_cercano(xx[k],yy[k],xx[kp1],yy[kp1],ppx,ppy)
                idd = np.rint(dd)
                dist = np.sqrt((ppx - p1)**2 + (ppy - p2)**2)
                if dc_host[ninn] == -1:
                   dc_host[ninn] = dist
                   ic_host[ninn] = k+idd
             else:
                ninn = iii+(jjj-1)*n[0]
                ii_host[ninn] = -1
                ppx = float(iii)*dx[0]
                ppy = float(jjj-1)*dx[1]
                dd,p1,p2 = punto_cercano(xx[k],yy[k],xx[kp1],yy[kp1],ppx,ppy)
                idd = np.rint(dd)
                dist = np.sqrt((ppx - p1)**2 + (ppy - p2)**2)
                if dc_host[ninn] == -1:
                   dc_host[ninn] = dist
                   ic_host[ninn] = k+idd
         if v1==-1 and v2==-1:
             if ii_host[inn] == -1:
                ninn = iii+1+(jjj-1)*n[0]
                ii_host[ninn] = 1
                ppx = float(iii+1)*dx[0]
                ppy = float(jjj-1)*dx[1]
                dd,p1,p2 = punto_cercano(xx[k],yy[k],xx[kp1],yy[kp1],ppx,ppy)
                idd = np.rint(dd)
                dist = np.sqrt((ppx - p1)**2 + (ppy - p2)**2)
                if dc_host[ninn] == -1:
                   dc_host[ninn] = dist
                   ic_host[ninn] = k+idd
             else:
                ninn = iii-1+(jjj+1)*n[0]
                ii_host[ninn] = -1
                ppx = float(iii-1)*dx[0]
                ppy = float(jjj+1)*dx[1]
                dd,p1,p2 = punto_cercano(xx[k],yy[k],xx[kp1],yy[kp1],ppx,ppy)
                idd = np.rint(dd)
                dist = np.sqrt((ppx - p1)**2 + (ppy - p2)**2)
                if dc_host[ninn] == -1:
                   dc_host[ninn] = dist
                   ic_host[ninn] = k+idd


    ttp += numt[ic]
   
   return ii_host, ic_host


#======================
#  PROGRAMA PRINCIPAL
#======================

#=========================
#  PARÁMETROS PRIMARIOS
#=========================
# Número de celdas
n = np.array([513,513],dtype=np.int32)
# Tamaño del dominio (menor que uno) 
L = np.array([1.0,1.0],dtype=np.float64) 
# Frontera inmersa
n_curvas:np.int32 = 2
# Hilos por bloque
TILE_I:np.int32 = 32
TILE_J:np.int32 = 16
# Condición de frontera externa (0 = Dirichlet, 1 = Neumann) 
bc:np.int32 = 0

#=========================
#  PARÁMETROS DERIVADOS
#=========================
# Tamaño de las celdas
dx:float64 = (L/n)
print("dx = ",dx[0],dx[1])
udx2 = 1.0/(dx*dx)
# Paso de tiempo
dt:np.float64 = 0.25*(min(dx[0],dx[1])**2)
print("dt = ",dt)
# Total de puntos 
nt = n[0]*n[1]
print("puntos = ",n[0],"x",n[1],"=",nt)

#===========
#  INICIO
#==========
start = time.time()

#=======================================
#  En caso de tener fronteras inmersas 
#=======================================
#----------------------------
#  Leer e interpolar puntos 
#----------------------------
print("Leyendo puntos de fronteras inmersas...")
npuntos, numt, x_curva, y_curva = leer_curva(n_curvas,dx) 

#------------------------
#  Gráfica de las curvas
#------------------------
#plt.plot(x_curva,y_curva,'o',color='black')
#plt.show()

#--------------------------------------------------------
#  Marcar celdas por donde pasan las fronteras inmersas 
#--------------------------------------------------------
print("Marcando celdas vecinas a las curvas...")
ii_h, ic_h = marcar_malla(n,n_curvas,numt,x_curva,y_curva)

#----------------------
#  Arreglos en el CPU
#----------------------
salto_u_h = np.ones(sum(numt),dtype=np.float64)
salto_un_h = np.zeros(sum(numt),dtype=np.float64)

#----------------------
#  Arreglos en el GPU
#----------------------
ju_d = cuda.to_device(salto_u_h)
jun_d = cuda.to_device(salto_un_h)
xc_d = cuda.to_device(x_curva)
yc_d = cuda.to_device(y_curva)
ii_d = cuda.to_device(ii_h)
ic_d = cuda.to_device(ic_h)
icn_d = cuda.to_device(ic_h)

#=======================
# Arreglos en el CPU 
#=======================
u_h   = np.zeros(nt,dtype=np.float64)
rhs_h = np.zeros(nt,dtype=np.float64)
e_h   = np.zeros(nt,dtype=np.float64)

#===================
# Arreglos al GPU
#===================
u_d   = cuda.to_device(u_h) 
un_d  = cuda.to_device(u_h)
uu_d  = cuda.to_device(u_h)
e_d   = cuda.to_device(e_h)
en_d  = cuda.to_device(e_h)
res_d = cuda.to_device(e_h)
z_d   = cuda.to_device(e_h)

#========================
#  Malla y lado derecho 
#========================
X,Y = np.meshgrid(np.linspace(0.0,1.0,n[0],dtype=np.float64), \
                  np.linspace(0.0,1.0,n[1],dtype=np.float64))
rhs_h = np.reshape(100.0*(((1.0-(6.0*X*X))*Y*Y*(1.0-(Y*Y)))   \
             +((1.0-(6.0*Y*Y))*X*X*(1.0-(X*X)))),(n[0]*n[1],))
rhsr_h = rhs_h
rhs_d = cuda.to_device(rhs_h)
rhsn_d = cuda.to_device(e_h)
rhsr_d = cuda.to_device(rhsr_h)

#=========================
#  Resolver la ecuación
#=========================
Multilevel(bc,n[0],n[1],res_d,e_h,rhs_d,rhsr_d,dx[0],dx[1],dt, \
           uu_d,e_d,un_d,en_d,rhsn_d,rhs_h,u_d,z_d,ii_d,ic_d,  \
           xc_d,yc_d,ju_d,jun_d)

#=========================
# Pasar arreglos al CPU
#=========================
u_d.copy_to_host(u_h)
#u_h = [float(ii_h[i]) for i in range(len(ic_h))]
end = time.time()
print("Tardó: ",end-start,"s")

#==================
# Graficar en 3D
#==================
u_h = np.reshape(u_h,(n[0],n[1]))
x,y = np.meshgrid(np.arange(0,L[0],dx[0]),np.arange(0,L[1],dx[1]))
#ax = plt.axes(projection='3d')
#ax.plot_surface(x,y,u_h,cmap=cm.hsv)
plt.contourf(x,y,u_h,64,cmap=cm.inferno)
plt.show()

#==============
# Archivo VTK  
#==============
u_h = np.reshape(u_h.astype(float),(n[0],n[1],1))
x,y,z = np.meshgrid(np.arange(0,L[0],dx[0]),np.arange(0,L[1],dx[1]),0)
gridToVTK("./solucion", x, y, z, pointData = {"solución":u_h})
