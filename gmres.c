//--------------------------------------//
// Generalized Minimal RESidual method  //
// (para arreglos)                      //
// gcc -o gmres gmres.c -lm -O3
//--------------------------------------//
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#define Real double
#define NUP 500

void GMRES(Real Wmat[NUP][NUP], Real bvec[NUP], Real *x, Real tol, int np)
{ int const nhu = NUP; int const nhup1 = NUP+1;
  int i,j,K,I,IQ,KP1,KB,LL,LLp1,maxite;
  Real v[NUP][NUP+1],bnorm,prod,rho,snormw;
  Real hes[NUP+1][NUP],q[2*NUP],r02[NUP+1];
  Real dwrk[NUP],dnw;
  Real vnrm,tem,sumdsq,arg,T1,T2,C,S,T;

  maxite=nhu-1;

  // First vector is b
  for (i=0;i<np;i++)  v[i][0]=bvec[i];
  dnw = 0.f; for (i=0;i<np;i++) dnw += bvec[i]*bvec[i]; bnorm = sqrt(dnw);
  for (i=0;i<np;i++) v[i][0] *= 1.f/bnorm;
  for(j=0;j<np;j++) for (i=0;i<np;i++) { hes[i][j] = 0.f;}
  prod=1.f;

  // Arnoldi iterations
  for (LL=0;LL<nhu;LL++) 
  { 
    LLp1=LL+1;

    // Matrix times vector (Krylov space)
    for (j=0;j<np;j++) {
     for (i=0;i<np;i++) {
     dwrk[i]=Wmat[j][i]*v[i][LL];
     }
     v[j][LLp1]= 0.f;
     for (i=0;i<np;i++) {
     v[j][LLp1] += dwrk[i];
     }
    }

    // Forming new orthogonal vector q and new column in Hessenberg
    dnw = 0.f; for (i=0;i<np;i++) dnw += v[i][LLp1]*v[i][LLp1]; 
    vnrm = sqrt(dnw);

    for (i=0;i<LL+1;i++) {
     hes[i][LL]=0.f;
     for (j=0;j<np;j++) hes[i][LL] += v[j][i]*v[j][LLp1];
     tem=-hes[i][LL];
     for (j=0;j<np;j++) v[j][LLp1] += tem*v[j][i];
    }
    dnw = 0.f; for (i=0;i<np;i++) dnw += v[i][LLp1]*v[i][LLp1]; 
    snormw = sqrt(dnw);

    if (vnrm+0.001f*snormw == vnrm) { 
      sumdsq=0.f;
      for (i=0;i<LL+1;i++) {
       tem = 0.f; for (j=0;j<np;j++) tem +=-v[j][i]*v[j][LLp1];
       if (hes[i][LL]+0.001f*tem != hes[i][LL]) { 
        hes[i][LL]+=-tem;
        for (j=0;j<np;j++) v[j][LLp1] += tem*v[j][i];
        sumdsq+=tem*tem;
       } 
      } 
      if (sumdsq != 0.0f) {
       arg=snormw*snormw-sumdsq;
       if (arg < 0.f) arg = 0.f; 
       snormw = sqrt(arg);
      }
    }

    // Last element of Hessenberg
    hes[LLp1][LL]=snormw;

    // Scale q_(n+1) 
    for (i=0;i<np;i++) v[i][LLp1] *= 1.f/snormw;

    // Givens rotations
    for(K=0; K<LL; K++) {
     I=2*K; 
     T1=hes[K][LL]; T2=hes[K+1][LL]; 
     C=q[I]; S=q[I+1]; 
     hes[K][LL]=C*T1-S*T2; hes[K+1][LL]=S*T1+C*T2;
    }
    T1=hes[LL][LL]; T2=hes[LL+1][LL];
    if (T2==0.f) { C=1.0f; S=0.f; }
    else if (fabs(T2) >= fabs(T1)) {
     T=T1/T2; S=-1.0f/sqrt(1.0f+T*T); C=-S*T;
    } else { 
     T=T2/T1; C= 1.0f/sqrt(1.0f+T*T); S=-C*T;
    }
    IQ=2*LL; q[IQ]=C; q[IQ+1]=S;
    hes[LL][LL]=C*T1-S*T2;

    // Residual update 
    prod*=q[2*LL+1];
    rho = prod;
    if (prod < 0) rho = - prod;

    // If residual reached tolerance we are ready
    if ((rho<=tol)||(LL==maxite-1)) { 
     for (j=0;j<nhup1;j++) r02[j] = 0.f;

     // Vector y formed (called r02)
     r02[0]=bnorm;

     for (K=0;K<LL+1;K++) {
      KP1=K+1; IQ=2*K;
      C=q[IQ]; S=q[IQ+1];
      T1=r02[K]; T2=r02[KP1];
      r02[K]=C*T1-S*T2; r02[KP1]=S*T1+C*T2;
     } 
     for (KB=0;KB<LL+1;KB++) {
      K=LL-KB; r02[K]=r02[K]/hes[K][K];
      T=-r02[K];
      for (i=0;i<K;i++) r02[i] += T*hes[i][K];
     }    
     // Solution x computed as linear combination of basis functions 
     for (j=0;j<np;j++) x[j] = 0.f;
     for (i=0;i<LL+1;i++) {
      for (j=0;j<np;j++) {
       x[j] += r02[i]*v[j][i];}
     }
     //printf("%d \t %17.17f \n",LL+1,rho);
     return;
    } 
    //printf("%d \t %17.17f \n",LL+1,rho);
    
  } 
  //printf("nhu too small \n");
  return;
} 

int main(int argc, char *argv[])
{
  int i,j,n;
  Real b[NUP],A[NUP][NUP],tol,xexact[NUP],dn1,dn2,dnw;
  Real xdiff[NUP],dwrk[NUP];
  Real *x;
  x = (Real *)malloc(sizeof(Real)*NUP);
  n = NUP; tol = 0.0001f;
  printf ("Equations and tolerance: %d\t%e\n",n,tol);
  srand48(time(0));
  for(i=0;i<n;i++) {
   for (j=0;j<n;j++) {
     A[i][j] = 2.f*drand48()-1.f;
   }
  xexact[i]=drand48();
  }
  for (j=0;j<n;j++) {
   for (i=0;i<n;i++) {
     dwrk[i]=A[j][i]*xexact[i];
   }
  b[j]= 0.f;
   for (i=0;i<n;i++) {
     b[j] += dwrk[i];
   }
  }

  //printf("setup done\n");

  Real time1 = clock();
  GMRES(A,b,x,tol,n);
  Real time2 = clock();

  printf ("tiempo = %e sec\n",(time2-time1)/CLOCKS_PER_SEC); 

  for (i=0;i<n; i++)  xdiff[i]=x[i]-xexact[i];
  dnw = 0.f; for (i=0;i<n; i++) dnw += xdiff[i]*xdiff[i];
  dn1 = sqrt(dnw);
  dnw = 0.f; for (i=0;i<n; i++) dnw += xexact[i]*xexact[i];
  dn2 = sqrt(dnw);
  printf("relative error %17.17f\n",dn1/dn2);
  free(x);
}
