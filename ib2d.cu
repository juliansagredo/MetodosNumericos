//-------------------------------------------------------------------
//  Geometric Single Grid Multilevel 
//  Immersed Boundary
//  Abril 2018
//-------------------------------------------------------------------
//  Autores: Julian T. Becerra Sagredo
//  nvcc -o ib2d ib2d.cu
//-------------------------------------------------------------------

//------------
// Librerias
//------------
#include <stdio.h>
#include <stdlib.h>

//--------------------
// Tamaño del bloque
//--------------------
#define TILE_I 32
#define TILE_J 32
#define TILE_P 32

//----------------------
// Precision de reales
//----------------------
#define Real double 

//---------------------
// Punteros en el CPU 
//---------------------
Real *rhs_host, *t_host, *res_host;
Real *tt_host, *err_host;
Real *rres_host;
Real *s_host;
Real *x_curva, *y_curva, *xx, *yy, *tt;
Real *larc;
int *ii_host, *ic_host, *icn_host;
Real *ju_host, *jun_host;

//--------------------------------
// Punteros y arreglos en el GPU
//--------------------------------
Real *rhs_data, *t_data, *err_data;
Real *tt_data, *rhs_array;
Real *t_array, *err_array;
Real *res_data, *rhsr_data, *rhs_mesh;
Real *s_data;
Real *xc_data, *yc_data, *tc_data;
int *ii_data, *ic_data, *icn_data;
Real *ju_data, *jun_data;

//---------------------
// Escalares globales
//---------------------
int ni,nj,nc,numtt,inm,tpoints;
Real li,lj,dx,dy,dt,kcond;

//----------------------
// Archivos de lectura 
//----------------------
FILE *fp_curva[2];

//--------------------------
// Definicion CUDA kernels 
//--------------------------
__global__ void suma_kernel (int ni, int nj, Real *t_data, Real *tt_data);

__global__ void restrict_kernel (int ll, int ni, int nj, Real *rhsr_data, 
		                 Real *rhs_array);

__global__ void relax_kernel (int ll, int ni, int nj, Real kcond, Real dt,
                              Real dx, Real dy, Real *err_data, Real *t_data,
                              Real *rhs_data, Real *res_data, Real *t_array,
                              Real *err_array, int resflag);

__global__ void inmersa_kernel (int l, int ni, int nj, int numtt, Real dx, Real dy, 
		                int *ii_data, int *ic_data, int *icn_data, 
				Real *xc_data, Real *yc_data, Real *t_array, 
				Real *rhs_data, Real *ju_data, Real *jun_data);

//------------------------
// Definicion C wrappers
//------------------------
void relax(int ll, Real dx, Real dy, Real kcond, Real dt, int ni, int nj, 
           Real *t_data, Real *err_data, Real *t_array, Real *err_array, 
	   Real *rhs_data, Real *res_data, int resflag);
void restrict(int ll, int ni, int nj, Real *rhs_array, Real *rhsr_data);
int  power(int a, int b);
void Inicia(int argc, char *argv[]);
void Multilevel(Real *rhs_host);
void apply_BCs(int ni, int nj, Real *t_data, Real *err_data);
void suma(int ni, int nj, Real *t_data, Real *tt_data);
void imprimir(int paso);
void curva(void);
void inmersa(int l); 

///////////////////////////////////////////////////////////////////
//   MAIN
///////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
    // n debe ser 2^m + 1
    ni = 513;
    nj = 513;
    nc = 2;
    li = 1.0;
    lj = 1.0;
    dx = li/Real(ni-1);
    dy = dx;
    //dy = lj/Real(nj-1);
    kcond = 0.01;
    dt = 0.25f*dx*dx/kcond;

    printf ("ni = %d\n", ni);
    printf ("nj = %d\n", nj);
    printf ("Número de puntos = %d\n", ni*nj);
    printf ("Número de curvas = %d\n", nc);

    inm = 1; 
    if (inm == 1) printf ("Con fronteras inmersas \n");

    Real time1 = clock();
    Inicia(argc, argv);
    Real time2 = clock();
    Multilevel(rhs_host);
    cudaMemcpy(s_data, tt_data, sizeof(Real)*ni*nj,cudaMemcpyDeviceToDevice);
    Real time3 = clock();

    printf ("tiempo = %e sec\n",(time3-time2)/CLOCKS_PER_SEC);

    return 0;
}

//-----------
// Inicia
//-----------
void Inicia(int argc, char** argv)
{
    
    int ntot = ni*nj;

    // Aparta la memoria en el CPU (host)
    t_host    = (Real *)malloc(ntot*sizeof(Real));
    tt_host   = (Real *)malloc(ntot*sizeof(Real));
    res_host  = (Real *)malloc(ntot*sizeof(Real));
    rres_host = (Real *)malloc(ntot*sizeof(Real));
    rhs_host  = (Real *)malloc(ntot*sizeof(Real));
    err_host  = (Real *)malloc(ntot*sizeof(Real));
    s_host    = (Real *)malloc(ntot*sizeof(Real));
    ii_host   = (int  *)malloc(ntot*sizeof(int));
    ic_host   = (int  *)malloc(ntot*sizeof(int)); 

    // Aparta la memoria en el GPU (device)
    cudaMalloc((void **)&t_data,   sizeof(Real)*ntot);
    cudaMalloc((void **)&tt_data,  sizeof(Real)*ntot);
    cudaMalloc((void **)&t_array,  sizeof(Real)*ntot);
    cudaMalloc((void **)&rhs_data, sizeof(Real)*ntot);
    cudaMalloc((void **)&rhs_array,sizeof(Real)*ntot);
    cudaMalloc((void **)&rhsr_data,sizeof(Real)*ntot);
    cudaMalloc((void **)&rhs_mesh, sizeof(Real)*ntot);
    cudaMalloc((void **)&err_data, sizeof(Real)*ntot);
    cudaMalloc((void **)&err_array,sizeof(Real)*ntot);
    cudaMalloc((void **)&res_data, sizeof(Real)*ntot);
    cudaMalloc((void **)&s_data,   sizeof(Real)*ntot);
    cudaMalloc((void **)&ii_data,  sizeof(int)*ntot);
    cudaMalloc((void **)&ic_data,  sizeof(int)*ntot);
    
    //-------------------
    // Valores iniciales    
    //-------------------
    for (int j=0; j<nj; j++)
     for (int i=0; i<ni; i++){
       int i2d = i + j*ni;
       t_host[i2d] = 0.0f;
       tt_host[i2d] = 0.0f;
       res_host[i2d] = 0.0f;
       rres_host[i2d] = 0.0f;
       err_host[i2d] = 0.0f;
       rhs_host[i2d] = 0.0f;
       s_host[i2d] = 1.f;
       ii_host[i2d] = 0;
       ic_host[i2d] = 0;
     }

     Real xi0 = 0.f;
     Real yj0 = 0.f;
     for (int k=0; k<1; k++){
      int ii = 0;
      for (int j=0; j<nj; j++){
       for (int i=0; i<ni; i++){
        Real xi = xi0 + Real(i)*dx;
        Real yj = yj0 + Real(j)*dy;
        rhs_host[ii] = 100.f*(((1.f-(6.f*xi*xi))*yj*yj*(1.f-(yj*yj))) 
                         +((1.f-(6.f*yj*yj))*xi*xi*(1.f-(xi*xi))));
	//rhs_host[ii] = 1.f*exp( -((xi-0.5f)*(xi-0.5f)+(yj-0.5f)*(yj-0.5f))/0.01f);
        ii++;
        }
       }
     }

    //------------------------
    // Leer la curva inmersa 
    //------------------------
    if (inm == 1) curva(); 

    // Memoria para la curva en GPU
    cudaMalloc((void **)&xc_data,  sizeof(Real)*numtt);
    cudaMalloc((void **)&yc_data,  sizeof(Real)*numtt);
    cudaMalloc((void **)&tc_data,  sizeof(Real)*numtt);
    
    // Copia valores iniciales al GPU
    cudaMemcpy(t_data,  t_host, sizeof(Real)*ntot,cudaMemcpyHostToDevice);
    cudaMemcpy(tt_data, tt_host, sizeof(Real)*ntot,cudaMemcpyHostToDevice);
    cudaMemcpy(t_array, t_host, sizeof(Real)*ntot,cudaMemcpyHostToDevice);
    cudaMemcpy(rhs_mesh,  rhs_host, sizeof(Real)*ntot,cudaMemcpyHostToDevice);
    cudaMemcpy(rhs_data,  rhs_host, sizeof(Real)*ntot,cudaMemcpyHostToDevice);
    cudaMemcpy(rhs_array, rhs_host, sizeof(Real)*ntot,cudaMemcpyHostToDevice);
    cudaMemcpy(err_data,  err_host, sizeof(Real)*ntot,cudaMemcpyHostToDevice);
    cudaMemcpy(err_array, err_host, sizeof(Real)*ntot,cudaMemcpyHostToDevice);
    cudaMemcpy(res_data,  res_host, sizeof(Real)*ntot,cudaMemcpyHostToDevice);
    cudaMemcpy(s_data,    s_host,   sizeof(Real)*ntot,cudaMemcpyHostToDevice);
    
    if (inm == 1) {
       cudaMemcpy(xc_data, xx, sizeof(Real)*numtt,cudaMemcpyHostToDevice);
       cudaMemcpy(yc_data, yy, sizeof(Real)*numtt,cudaMemcpyHostToDevice);
       cudaMemcpy(tc_data, tt, sizeof(Real)*numtt,cudaMemcpyHostToDevice);
       cudaMemcpy(ii_data, ii_host, sizeof(int)*ntot,cudaMemcpyHostToDevice);
       cudaMemcpy(ic_data, ic_host, sizeof(int)*ntot,cudaMemcpyHostToDevice);
    
       cudaMalloc((void **)&icn_data, sizeof(int)*ntot);
       cudaMalloc((void **)&ju_data,  sizeof(Real)*tpoints);
       cudaMalloc((void **)&jun_data, sizeof(Real)*tpoints);

       cudaMemcpy(icn_data, icn_host, sizeof(int)*ntot,cudaMemcpyHostToDevice);
       cudaMemcpy(ju_data, ju_host, sizeof(Real)*tpoints,cudaMemcpyHostToDevice);
       cudaMemcpy(jun_data, jun_host, sizeof(Real)*tpoints,cudaMemcpyHostToDevice);
     }
    
}

//-------------------
// Frontera inmersa 
//-------------------
void curva(void)
{
  //---------------------------
  // Leer datos de las curvas  
  //---------------------------
  char c_file[40];
  int npuntos[nc];
  int tp = 0;
  float scalecx[nc],scalecy[nc],xc0[nc],yc0[nc];

  for (int ic=0; ic<nc; ic++) { 
    sprintf(c_file,"./PUNTOS/curva%d.txt",ic);
    fp_curva[ic] = fopen(c_file,"r");
    if (fp_curva[ic]==NULL) printf("Error: can't open file \n");
    fscanf (fp_curva[ic], "%d%f%f%f%f",&npuntos[ic],&scalecx[ic],&scalecy[ic],&xc0[ic],&yc0[ic]);
    printf ("npuntos = %d\n", npuntos[ic]);
    tp += npuntos[ic];
    fclose(fp_curva[ic]);
  }

  printf("tp = %d\n",tp); 

  x_curva = (Real *)malloc(tp*sizeof(Real));
  y_curva = (Real *)malloc(tp*sizeof(Real));

  int l0=2;
  int ll[tp];
  tp = 0;
  for (int ic=0; ic<nc; ic++) {
    sprintf(c_file,"./PUNTOS/curva%d.txt",ic);
    fp_curva[ic] = fopen(c_file,"r");
    fscanf (fp_curva[ic], "%d%f%f%f%f",&npuntos[ic],&scalecx[ic],&scalecy[ic],&xc0[ic],&yc0[ic]);
    for (int icurv=0; icurv<npuntos[ic]; icurv++){
      int lcont;
      float xc, yc;
      fscanf(fp_curva[ic], "%d%f%f", &lcont, &xc, &yc);
      x_curva[tp+icurv] = Real(xc0[ic])+Real(scalecx[ic])*Real(xc);
      y_curva[tp+icurv] = Real(yc0[ic])+Real(scalecy[ic])*Real(yc);
      ll[tp+icurv] = lcont;
    }
    tp += npuntos[ic];
    fclose(fp_curva[ic]);
  }

  tp = 0;
  int numt[nc];
  numtt = 0;
  Real ds[nc];
  Real dxx[nc];
  for (int ic=0; ic<nc; ic++) {
    ds[ic] = sqrt((x_curva[tp+0]-x_curva[tp+1])*(x_curva[tp+0]-x_curva[tp+1])
                +(y_curva[tp+0]-y_curva[tp+1])*(y_curva[tp+0]-y_curva[tp+1]));
    Real alphadx = 0.1;
    numt[ic] = int(Real(npuntos[ic])*ds[ic]/(alphadx*dx));
    dxx[ic] = Real(npuntos[ic])*ds[ic]/Real(numt[ic]);
    numtt += numt[ic];
    tp += npuntos[ic];
  }

  larc = (Real *)malloc(numtt*sizeof(Real));
  xx   = (Real *)malloc(numtt*sizeof(Real));
  yy   = (Real *)malloc(numtt*sizeof(Real));
  tt   = (Real *)malloc(numtt*sizeof(Real));

  tp = 0;
  int ttp = 0;
  for (int ic=0; ic<nc; ic++) {
    //------------------------------------------------------------------
    // Generar puntos sobre la curva con longitud de arco y Z0, Z1 o Z2
    //------------------------------------------------------------------
    Real xxx0,xx1,xx2,xx3,xx4,xx5;
    Real a0,a1,a2,a3,a4,a5;
    Real ax0,ax1,ax2,ax3,ax4,ax5;
    Real ay0,ay1,ay2,ay3,ay4,ay5;


    for (int iip = 0; iip<numt[ic]; iip++) larc[ttp+iip] = Real(iip)*(dxx[ic]/ds[ic]);
	                                  

    for (int ipunt = 0; ipunt<numt[ic]; ipunt++) {
      int l = l0;
      int idx = int(larc[ttp+ipunt]);

      if (idx>0 && idx < npuntos[ic]-2) {
        if (ll[tp+idx-1] == 0 || ll[tp+idx+2]==0) l = 1;}
      else if (idx == 0) {
        if (ll[tp+npuntos[ic]-1] == 0 || ll[tp+idx+2]==0) l=1;}
      else if (idx == npuntos[ic]-2) {
        if (ll[tp+npuntos[ic]-3] == 0 || ll[tp+0] == 0) l=1;}
      else if (idx == npuntos[ic]-1) {
        if (ll[tp+npuntos[ic]-2] == 0 || ll[tp+1] == 0) l=1;}
      else if (idx == npuntos[ic]) {
        if (ll[tp+npuntos[ic]-1] == 0 || ll[tp+2] == 0) l=1;}

      if (idx<npuntos[ic]-1){
        if (ll[tp+idx] == 0 || ll[tp+idx+1] == 0) l = 0;}
      else {
        if (ll[tp+npuntos[ic]-1]==0 || ll[tp+0] ==0) l = 0;}

      //---------------------------
      // Interpolacion quintica Z2
      //---------------------------
      if(l==2) {

        xx2 = larc[ttp+ipunt]-Real(idx);
        xxx0 = xx2+2.f;
        xx1 = xx2+1.f;
        xx3 = 1.f-xx2;
        xx4 = 2.f-xx2;
        xx5 = 3.f-xx2;
        a0 = 18.f+xxx0*((-153.f/4.f)+xxx0*((255.f/8.f)+xxx0*((-313.f/24.f)+xxx0*((21.f/8.f)+(-5.f/24.f)*xxx0))));
        a1 = -4.f+xx1*((75.f/4.f)+xx1*((-245.f/8.f) + xx1*((545.f/24.f)+xx1*((-63.f/8.f)+xx1*(25.f/24.f)))));
        a2 = 1.f+xx2*xx2*((-15.f/12.f)+xx2*((-35.f/12.f)+xx2*((63.f/12.f)+xx2*(-25.f/12.f))));
        a3 = 1.f+xx3*xx3*((-15.f/12.f)+xx3*((-35.f/12.f)+xx3*((63.f/12.f)+xx3*(-25.f/12.f))));
        a4 = -4.f+xx4*((75.f/4.f)+xx4*((-245.f/8.f)+xx4*((545.f/24.f)+xx4*((-63.f/8.f)+xx4*(25.f/24.f)))));
        a5 = 18.f+xx5*((-153.f/4.f)+xx5*((255.f/8.f)+xx5*((-313.f/24.f)+xx5*((21.f/8.f)+(-5.f/24.f)*xx5))));

        if (idx<npuntos[ic]) { ax2 = x_curva[tp+idx];} else {ax2 = x_curva[tp+0]; }
        if (idx<npuntos[ic]) { ay2 = y_curva[tp+idx];} else {ay2 = y_curva[tp+0]; }

        if (idx>1) {ax0 = x_curva[tp+idx-2];} else if (idx>0) {ax0=x_curva[tp+npuntos[ic]-1];}
        else {ax0=x_curva[tp+npuntos[ic]-2];}
        if (idx>0) {ax1 = x_curva[tp+idx-1];} else {ax1 = x_curva[tp+npuntos[ic]-1];}
        if (idx<npuntos[ic]-1) {ax3 = x_curva[tp+idx+1];} else {ax3 = x_curva[tp+0];}
        if (idx<npuntos[ic]-2) {ax4 = x_curva[tp+idx+2];} else if (idx<npuntos[ic]-1) {ax4 = x_curva[tp+0];}
        else {ax4=x_curva[1];}
        if (idx<npuntos[ic]-3) {ax5 = x_curva[tp+idx+3];} else if (idx<npuntos[ic]-2) {ax5 = x_curva[tp+0];}
        else if (idx<npuntos[ic]-1) {ax5 = x_curva[tp+1];} else {ax5 = x_curva[tp+2]; }
        if (idx>0) {ay1 = y_curva[tp+idx-1];} else {ay1 = y_curva[tp+npuntos[ic]-1];}
        if (idx>1) {ay0 = y_curva[tp+idx-2];} else if(idx>0) {ay0=y_curva[tp+npuntos[ic]-1];}
        else {ay0=y_curva[tp+npuntos[ic]-2];}
        if (idx<npuntos[ic]-1) {ay3 = y_curva[tp+idx+1];} else {ay3 = y_curva[tp+0];}
        if (idx<npuntos[ic]-2) {ay4 = y_curva[tp+idx+2];} else if(idx<npuntos[ic]-1) {ay4 = y_curva[tp+0];}
        else {ay4=y_curva[tp+1];}
        if (idx<npuntos[ic]-3) { ay5 = y_curva[tp+idx+3];} else if (idx<npuntos[ic]-2) {ay5 = y_curva[tp+0];}
        else if (idx<npuntos[ic]-1) {ay5 = y_curva[tp+1];} else {ay5 = y_curva[tp+2];}

        xx[ttp+ipunt] =  a0*ax0+a1*ax1+a2*ax2+a3*ax3+a4*ax4+a5*ax5;
        yy[ttp+ipunt] =  a0*ay0+a1*ay1+a2*ay2+a3*ay3+a4*ay4+a5*ay5;
      }

      //--------------------------
      // Interpolacion cubica Z1
      //--------------------------
      if(l==1){

        xx1 = larc[ttp+ipunt]-Real(idx);
        xxx0 = xx1+1.f;
        xx2 = 1.f-xx1;
        xx3 = 2.f-xx1;
        a0 = 0.5f*(2.f-xxx0)*(2.f-xxx0)*(1.f-xxx0);
        a1 = 1.f-2.5f*(xx1*xx1)+1.5f*(xx1*xx1*xx1);
        a2 = 1.f-2.5f*(xx2*xx2)+1.5f*(xx2*xx2*xx2);
        a3 = 0.5f*(2.f-xx3)*(2.f-xx3)*(1.f-xx3);

        if (idx>0) {ax0 = x_curva[tp+idx-1];} else {ax0 = x_curva[tp+npuntos[ic]-1];}
        if (idx<npuntos[ic]) {ax1 = x_curva[tp+idx];} else {ax1 = x_curva[tp+0];}
        if (idx<npuntos[ic]-1) {ax2 = x_curva[tp+idx+1];} else if (idx<npuntos[ic]) {ax2 = x_curva[tp+0];}
        else {ax2 = x_curva[tp+1];}
        if (idx<npuntos[ic]-2) {ax3 = x_curva[tp+idx+2];} else if (idx<npuntos[ic]-1) {ax3 = x_curva[tp+0];}
        else if (idx<npuntos[ic]) {ax3 = x_curva[tp+1];} else {ax3 = x_curva[tp+2];}
        if (idx>0) {ay0 = y_curva[tp+idx-1];} else {ay0 = y_curva[tp+npuntos[ic]-1];}
        if (idx<npuntos[ic]) {ay1 = y_curva[tp+idx];} else {ay1 = y_curva[tp+0];}
        if (idx<npuntos[ic]-1) {ay2 = y_curva[tp+idx+1];} else if (idx<npuntos[ic]) {ay2 = y_curva[tp+0];}
        else {ay2 = y_curva[tp+1];}
        if (idx<npuntos[ic]-2) {ay3 = y_curva[tp+idx+2];} else if (idx<npuntos[ic]-1) {ay3 = y_curva[tp+0];}
        else if (idx<npuntos[ic]) {ay3 = y_curva[tp+1];} else {ay3 = y_curva[tp+2];}

        xx[ttp+ipunt] =  a0*ax0+a1*ax1+a2*ax2+a3*ax3;
        yy[ttp+ipunt] =  a0*ay0+a1*ay1+a2*ay2+a3*ay3;
      }

      //--------------------------
      // Interpolacion lineal Z0 
      //--------------------------
      if(l==0){
        xxx0 = larc[ttp+ipunt]-Real(idx);
        xx1 = 1.f-xxx0;
        a0 = xx1;
        a1 = xxx0;
        if (idx<npuntos[ic]) {ax0 = x_curva[tp+idx];} else {ax0 = x_curva[tp+0];}
        if (idx<npuntos[ic]-1) {ax1 = x_curva[tp+idx+1];} else if (idx<npuntos[ic]) {ax1 = x_curva[tp+0];}
        else {ax1 = x_curva[tp+1];}
        if (idx<npuntos[ic]) {ay0 = y_curva[tp+idx];} else {ay0 = y_curva[tp+0];}
        if (idx<npuntos[ic]-1) {ay1 = y_curva[tp+idx+1];} else if (idx < npuntos[ic]) {ay1 = y_curva[tp+0];}
        else {ay1 = y_curva[tp+1];}

        xx[ttp+ipunt] =  a0*ax0+a1*ax1;
        yy[ttp+ipunt] =  a0*ay0+a1*ay1;
      }
      //printf("x,y = %d\t%e\t%e\n",ic,xx[ttp+ipunt],yy[ttp+ipunt]);
    }

    //-----------------------------------------
    // Marcar los vecinos de la curva inmersa 
    //-----------------------------------------
    Real pi = 3.141592654f;
    Real aa1 = 50.0f/(dx*dy);
    for (int k=ttp; k<ttp+numt[ic]; k++) {

      int km1 = k-1;
      int kp1 = k+1;

      if (k == ttp) km1 = ttp+numt[ic]-1;
      if (k == ttp+numt[ic]-1) kp1 = ttp;

      Real d1x = (xx[k] - xx[km1]);
      Real d2x = (xx[kp1]-xx[k]);
      Real d1y = (yy[k] - yy[km1]);
      Real d2y = (yy[kp1]-yy[k]);

      Real dxa = sqrtf(d1x*d1x + d1y*d1y);
      Real dxb = sqrtf(d2x*d2x + d2y*d2y);

      int iii = int(xx[k]/dx);
      int jjj = int(yy[k]/dy);
      int inn = iii+ni*jjj;
      int inn2 = iii+ni*(jjj+1);
      int inn3 = iii+ni*(jjj-1);
      int inn4 = iii+ni*(jjj+2);

      int tag1 = 1; int tag2 = 1; int tag3 = 1; int tag4 = 1;
      int tag5 = 1; int tag6 = 1; int tag7 = 1; int tag8 = 1;
      int tag9 = 1; int tag10= 1; int tag11= 1; int tag12= 1;

      Real Tx = 0.5f*(d2x+d1x);
      Real Ty = 0.5f*(d2y+d1y);
      Real normT = sqrt(Tx*Tx+Ty*Ty);
      Tx /= normT; 
      Ty /= normT;

      //-------------------------------------------------
      // Producto cruz entre tangente y puntos vecinos
      //-------------------------------------------------
      Real txx1 = Tx*(dy*Real(jjj)  -yy[k])-Ty*(dx*Real(iii)  -xx[k]);
      Real txx2 = Tx*(dy*Real(jjj)  -yy[k])-Ty*(dx*Real(iii+1)-xx[k]);
      Real txx3 = Tx*(dy*Real(jjj+1)-yy[k])-Ty*(dx*Real(iii)  -xx[k]);
      Real txx4 = Tx*(dy*Real(jjj+1)-yy[k])-Ty*(dx*Real(iii+1)-xx[k]);
      Real txx5 = Tx*(dy*Real(jjj)  -yy[k])-Ty*(dx*Real(iii-1)-xx[k]);
      Real txx6 = Tx*(dy*Real(jjj+1)-yy[k])-Ty*(dx*Real(iii-1)-xx[k]);
      Real txx7 = Tx*(dy*Real(jjj)  -yy[k])-Ty*(dx*Real(iii+2)-xx[k]);
      Real txx8 = Tx*(dy*Real(jjj+1)-yy[k])-Ty*(dx*Real(iii+2)-xx[k]);
      Real txx9 = Tx*(dy*Real(jjj-1)-yy[k])-Ty*(dx*Real(iii)  -xx[k]);
      Real txx10= Tx*(dy*Real(jjj-1)-yy[k])-Ty*(dx*Real(iii+1)-xx[k]);
      Real txx11= Tx*(dy*Real(jjj+2)-yy[k])-Ty*(dx*Real(iii)  -xx[k]);
      Real txx12= Tx*(dy*Real(jjj+2)-yy[k])-Ty*(dx*Real(iii+1)-xx[k]);
        
      if (txx1<0)  tag1=-1;  if (txx2<0)  tag2=-1;  
      if (txx3<0)  tag3=-1;  if (txx4<0)  tag4=-1;
      if (txx5<0)  tag5=-1;  if (txx6<0)  tag6=-1;  
      if (txx7<0)  tag7=-1;  if (txx8<0)  tag8=-1;
      if (txx9<0)  tag9=-1;  if (txx10<0) tag10=-1; 
      if (txx11<0) tag11=-1; if (txx12<0) tag12=-1;

      ii_host[inn]    = tag1;
      ii_host[inn+1]  = tag2;
      ii_host[inn2]   = tag3;
      ii_host[inn2+1] = tag4;
     
      ic_host[inn]    = k;
      ic_host[inn+1]  = k;
      ic_host[inn2]   = k;
      ic_host[inn2+1] = k;
   
      Real Tdx5 =  Ty*(dy*Real(jjj)  -yy[k]) + Tx*(dx*Real(iii-1)-xx[k]);
      Real Tdx6 =  Ty*(dy*Real(jjj+1)-yy[k]) + Tx*(dx*Real(iii-1)-xx[k]);
      Real Tdx7 =  Ty*(dy*Real(jjj)  -yy[k]) + Tx*(dx*Real(iii+2)-xx[k]);
      Real Tdx8 =  Ty*(dy*Real(jjj+1)-yy[k]) + Tx*(dx*Real(iii+2)-xx[k]);
      Real Tdx9 =  Ty*(dy*Real(jjj-1)-yy[k]) + Tx*(dx*Real(iii)  -xx[k]);
      Real Tdx10 = Ty*(dy*Real(jjj-1)-yy[k]) + Tx*(dx*Real(iii+1)-xx[k]);
      Real Tdx11 = Ty*(dy*Real(jjj+2)-yy[k]) + Tx*(dx*Real(iii)  -xx[k]);
      Real Tdx12 = Ty*(dy*Real(jjj+2)-yy[k]) + Tx*(dx*Real(iii+1)-xx[k]);

      Real deps = 0.1f*dx;

      if (abs(Tdx5) < deps) { ii_host[inn-1]  = tag5;  ic_host[inn-1] = k; } 
      if (abs(Tdx6) < deps) { ii_host[inn2-1] = tag6;  ic_host[inn2-1] = k;}
      if (abs(Tdx7) < deps) { ii_host[inn+2]  = tag7;  ic_host[inn+2] = k; }
      if (abs(Tdx8) < deps) { ii_host[inn2+2] = tag8;  ic_host[inn2+2] = k;}
      if (abs(Tdx9) < deps) { ii_host[inn3]   = tag9;  ic_host[inn3] = k;  }
      if (abs(Tdx10)< deps) { ii_host[inn3+1] = tag10; ic_host[inn3+1] = k;}
      if (abs(Tdx11)< deps) { ii_host[inn4]   = tag11; ic_host[inn4] = k;  }
      if (abs(Tdx12)< deps) { ii_host[inn4+1] = tag12; ic_host[inn4+1] = k;}

    }

    //-------------------
    //  Marcar esquinas 
    //-------------------
    Real rhstot = 0.0f;
    Real atot = 0.0f;
    int *iv;
    iv = (int *)malloc(9*sizeof(int));
    for (int j=1; j<nj-1; j++){
      for (int i=1; i<ni-1; i++){
        iv[0] = (i-1)+(j-1)*ni;
        iv[1] = (i)  +(j-1)*ni;
        iv[2] = (i+1)+(j-1)*ni;
        iv[3] = (i-1)+(j)*ni;
        iv[4] = (i)  +(j)*ni;
        iv[5] = (i+1)+(j)*ni;
        iv[6] = (i-1)+(j+1)*ni;
        iv[7] = (i)  +(j+1)*ni;
        iv[8] = (i+1)+(j+1)*ni;
        int flagiv = 0;
        for (int iiv=0; iiv<9; iiv++) {
          if (iiv!=4) {
            if (ii_host[iv[iiv]] == 1) { if (flagiv==-1) flagiv=2; else if (flagiv==2) flagiv=2; else flagiv= 1;} 
            if (ii_host[iv[iiv]] ==-1) { if (flagiv== 1) flagiv=2; else if (flagiv==2) flagiv=2; else flagiv=-1;} 
          }
        }
        if (flagiv == 2) { 
          int fiv = 0; int kref = 0; 
          if (ii_host[iv[1]] != 0) { fiv = ii_host[iv[1]]; kref = ic_host[iv[1]];} 
          if (ii_host[iv[3]] != 0) { fiv = ii_host[iv[3]]; kref = ic_host[iv[3]];}
          if (ii_host[iv[5]] != 0) { fiv = ii_host[iv[5]]; kref = ic_host[iv[5]];}
          if (ii_host[iv[7]] != 0) { fiv = ii_host[iv[7]]; kref = ic_host[iv[7]];}
          if (ii_host[iv[4]] == 0) { ii_host[iv[4]] = fiv; ic_host[iv[4]] = kref;}
        } 
        else { ii_host[iv[4]] = 0; ic_host[iv[4]] = 0;}
      } 
    }

    for (int j=1; j<nj-1; j++){
      for (int i=1; i<ni-1; i++){
        iv[0] = (i-1) + (j-1)*ni;
        iv[1] = (i)   + (j-1)*ni;
        iv[2] = (i+1) + (j-1)*ni;
        iv[3] = (i-1) + (j)*ni;
        iv[4] = (i)   + (j)*ni;
        iv[5] = (i+1) + (j)*ni;
        iv[6] = (i-1) + (j+1)*ni;
        iv[7] = (i)   + (j+1)*ni;
        iv[8] = (i+1) + (j+1)*ni;
        int flagiv = 0;
        for (int iiv=0; iiv<9; iiv++) {
          if (iiv!=4) {
            if (ii_host[iv[iiv]] == 1) { 
              if (flagiv==-1) flagiv=2; 
              else if (flagiv==2) flagiv=2; 
              else flagiv=1;}
            if (ii_host[iv[iiv]] ==-1) { 
              if (flagiv== 1) flagiv=2; 
              else if (flagiv==2) flagiv=2; 
              else flagiv=-1;}
          }
        }
        if (flagiv == 2) { 
 	 int fiv = 0; int kref = 0; 
          if (ii_host[iv[1]] != 0) { fiv = ii_host[iv[1]]; kref = ic_host[iv[1]];}
          if (ii_host[iv[3]] != 0) { fiv = ii_host[iv[3]]; kref = ic_host[iv[3]];}
          if (ii_host[iv[5]] != 0) { fiv = ii_host[iv[5]]; kref = ic_host[iv[5]];}
          if (ii_host[iv[7]] != 0) { fiv = ii_host[iv[7]]; kref = ic_host[iv[7]];}
          if (ii_host[iv[4]] == 0) { ii_host[iv[4]] = fiv; ic_host[iv[4]] = kref;}
        }
        else { ii_host[iv[4]] = 0; ic_host[iv[4]] = 0;}
      }
    }
    ttp += numt[ic];
    tp += npuntos[ic];
  }

  //-------------------------------------
  //  Punto más cercano sobre la curva 
  //-------------------------------------
  int kcount = 0;
  for (int j=1; j<nj-1; j++){
    for (int i=1; i<ni-1; i++){
      Real xm = Real(i)*dx;
      Real ym = Real(j)*dy;
      Real mindist = li;  
      int kk = ii_host[i+ni*j];
      if (kk != 0) {
        int k = ic_host[i+ni*j];
        int nbu = 30;
        int kmin = k;
	kcount +=1;
        for (int ik = -nbu; ik < nbu+1; ik++) {
          int kpik = k+ik;
	  if (kpik<0) kpik +=ttp;
	  if (kpik>ttp-1) kpik -=ttp;
          Real distancia = (xm-xx[kpik])*(xm-xx[kpik]) + (ym-yy[kpik])*(ym-yy[kpik]);
          if (distancia < mindist) { mindist = distancia; kmin = kpik;}
        }
        ic_host[i+ni*j]=kmin;
        //printf("index = %d\t%d\t%d\n",i,j,ic_host[i+ni*j]);
      }
      //printf("index = %d\t%d\t%d\t%d\n",i,j,ii_host[i+ni*j],ic_host[i+ni*j]);
    }
  }

  //------------------------------------------------
  //  Saltos sobre puntos más cercanos en la curva 
  //------------------------------------------------
  tpoints = kcount;
  ju_host  = (Real *)malloc(tpoints*sizeof(Real));
  jun_host = (Real *)malloc(tpoints*sizeof(Real));
  icn_host = (int *)malloc(   ni*nj*sizeof(int));
  for (int i=0; i<tpoints; i++){ 
    ju_host[i] = 1.f;
    jun_host[i] = 0.f;
  }

  //---------------------------------
  // Índices de puntos más cercanos 
  //---------------------------------
  kcount = 0;
  for (int j=1; j<nj-1; j++){
    for (int i=1; i<ni-1; i++){
      int kk = ii_host[i+ni*j];
      icn_host[i+ni*j] = 0;
      if (kk != 0) {
        int k = ic_host[i+ni*j];
        icn_host[i+ni*j]=kcount;
	kcount +=1;
        //printf("index = %d\t%d\t%d\n",i,j,icn_host[i+ni*j]);
      }
    }
  }


}


//-------------
// Multilevel
//-------------
void Multilevel(Real *rhs_host)
{
  int ntot = ni*nj;
  int nn; int nni; int nn2; int nnl;

  // cycle parameters 
  int npot = log2(Real(ni/2)); int nrel = ni; int nnrel = 1;
  int nvci = npot; int ndown = npot-2;
  int nmin = 2;
  int nnnn = 0;
  int cycles = 12;

  printf("ciclos = %d\n", cycles);
  printf("nmin = %d\n", nmin);

  cudaMemcpy(rhs_mesh, rhs_host, sizeof(Real)*ntot,cudaMemcpyHostToDevice);
  cudaMemcpy(rhs_data, rhs_mesh, sizeof(Real)*ntot,cudaMemcpyDeviceToDevice);

  //imprimir(nnnn);

  // residual equation iteration
  for (int ncyc = 0; ncyc<cycles; ncyc++) {

  ndown = npot-2;

   if (ncyc>0) {
    // residual 
    cudaMemcpy(rhs_mesh, res_data, sizeof(Real)*ntot, cudaMemcpyDeviceToDevice);
    cudaMemcpy(t_data,   err_host, sizeof(Real)*ntot, cudaMemcpyHostToDevice);
    //cudaMemcpy(res_data, err_host, sizeof(Real)*ntot, cudaMemcpyHostToDevice);
   }

   int count = 0;

   //-----------------------
   // Razor cycle for GPU 
   //-----------------------
   for (nni = 0; nni<nvci; nni++){

    // From coarsest down to fine
    for (nn2=ndown+2; nn2>ndown; nn2--){
     nnl = power(2,nn2);
     int nrelax = nrel/nnl;
     nrelax = min(nrelax,nmin);
     //nrelax = nmin;
     //printf("Nivel= %d\n ", nnl);
     cudaMemcpy(rhs_data, rhs_mesh, sizeof(Real)*ntot,cudaMemcpyDeviceToDevice);
     if (inm == 1 && ncyc==0) inmersa(0);
     cudaMemcpy(rhsr_data, rhs_data, sizeof(Real)*ntot, cudaMemcpyDeviceToDevice);
     for (int nn3 =0;nn3<nn2; nn3++) {
      int nn4 = power(2,nn3);
      restrict(nn4,ni,nj,rhs_array,rhsr_data);
      count += 1;
     }
 
     for (nn=0; nn<nrelax; nn++){
      relax(nnl,dx,dy,kcond,dt,ni,nj,t_data,err_data,t_array,err_array,rhsr_data,res_data,0);
      apply_BCs(ni, nj, t_data, err_data);
      count += 1;
      nnnn += 1;
      //imprimir(nnnn);
     }
    }
 
    // From fine to coarse
    for (nn2=ndown+2; nn2<ndown+3; nn2++){
     nnl = power(2,nn2);
     int nrelax = nrel/nnl;
     nrelax = min(nrelax,nmin);
     //nrelax = nmin;
     //printf("Nivel= %d\n ", nnl);
     cudaMemcpy(rhs_data, rhs_mesh, sizeof(Real)*ntot,cudaMemcpyDeviceToDevice);
     if (inm == 1 && ncyc==0) inmersa(0);
     cudaMemcpy(rhsr_data, rhs_data, sizeof(Real)*ntot, cudaMemcpyDeviceToDevice);
     for (int nn3 =0;nn3<nn2; nn3++) {
      int nn4 = power(2,nn3);
      restrict(nn4,ni,nj,rhs_array,rhsr_data);
      count +=1;
     }
     for (nn=0; nn<nrelax; nn++){
      relax(nnl,dx,dy,kcond,dt,ni,nj,t_data,err_data,t_array,err_array,rhsr_data,res_data,0);
      apply_BCs(ni, nj, t_data, err_data);
      count +=1;
      nnnn += 1;
      //imprimir(nnnn);
     }
    }
    ndown = ndown-1;
   }
 
   // From coarsest down to fine
   for (nni = 0; nni<nnrel; nni++){
    for (nn2=ndown+2; nn2>-1; nn2--){
     nnl = power(2,nn2);
     int nrelax = nrel/nnl;
     nrelax = min(nrelax,nmin);
     //nrelax = nmin;
     //printf("Nivel = %d\n ", nnl);
     cudaMemcpy(rhs_data, rhs_mesh, sizeof(Real)*ntot,cudaMemcpyDeviceToDevice);
     if (inm == 1 && ncyc==0) inmersa(0);
     cudaMemcpy(rhsr_data, rhs_data, sizeof(Real)*ntot, cudaMemcpyDeviceToDevice);
     for (int nn3 =0;nn3<nn2; nn3++) {
      int nn4 = power(2,nn3);
      restrict(nn4,ni,nj,rhs_array,rhsr_data);
      count +=1;
     }
     for (nn=0; nn<nrelax; nn++){
      relax(nnl,dx,dy,kcond,dt,ni,nj,t_data,err_data,t_array,err_array,rhsr_data,res_data,0);
      apply_BCs(ni, nj, t_data, err_data);
      count +=1;
      nnnn += 1;
      //imprimir(nnnn);
     }
    }
   }

   //printf("WU = %d\n",count);

     // pulido final
     for (nn=0; nn<1; nn++){
      cudaMemcpy(rhs_data, rhs_mesh, sizeof(Real)*ntot,cudaMemcpyDeviceToDevice);
      if (inm == 1 && ncyc==0) inmersa(0);
      relax(1,dx,dy,kcond,dt,ni,nj,t_data,err_data,t_array,err_array,rhsr_data,res_data,1);
      apply_BCs(ni, nj, t_data, err_data);
     }
      nnnn += 1;
      //imprimir(nnnn);

      // sumar a la solucion
      suma(ni,nj,t_data,tt_data);

      //imprimir(nnnn);
  }

      imprimir(nnnn);
      
}

int power( int a, int b)
{
 int c = a;
 for (int n=b; n>1; n--) c*= a;
 if (b == 0) c = 1;
 return c;
}

//////////////////////////////////////////////////////////////////////
/// IMPRIMIR
////////////////////////////////////////////////////////////////////// 
void imprimir(int paso)
{
    int ntot = ni*nj;
    char s_pars[40];
    FILE *fp;

    // hacemos la transferencia de datos al CPU:
    //cudaMemcpy(rhs_host, rhs_data, sizeof(Real)*ntot, cudaMemcpyDeviceToHost);
    cudaMemcpy(t_host, t_data, sizeof(Real)*ntot, cudaMemcpyDeviceToHost);
    cudaMemcpy(res_host, res_data, sizeof(Real)*ntot, cudaMemcpyDeviceToHost);
    cudaMemcpy(tt_host, tt_data, sizeof(Real)*ntot, cudaMemcpyDeviceToHost);

    Real errorL1 = 0.f; 
    Real umvh = 0.f; 
    Real utot = 0.f; 
    Real errorL2 = 0.f;
    Real errmax = 0.f;
    Real t_analitica = 0.f; 
    Real maxres = 0.f;
    Real normres = 0.f;
    Real maxres1 = 0.f;
    Real resL2 = 0.f;
    Real maxu = 0.f;
    Real mnormres;

    Real x1,x2;
    Real res = 0.0f;
    Real res1 = 0.f;
    for (int j=0;j<nj;j++) 
     for (int i=0;i<ni;i++){
        int i2d = i +j*ni;
        int i2d1 = i2d+1;
        int i2d2 = i2d-1;
        int i2d3 = i2d+ni;
        int i2d4 = i2d-ni;
        int i2d5 = i2d+1+ni;
        int i2d6 = i2d-1+ni;
        int i2d7 = i2d+1-ni;
        int i2d8 = i2d-1-ni;
        if (i == 0){i2d2 = i2d; i2d6 = i2d; i2d8 = i2d;}
        if (j == 0){i2d4 = i2d; i2d7 = i2d; i2d8 = i2d;}
        if (i == ni-1){i2d1 = i2d; i2d5 = i2d; i2d7 = i2d;}
        if (j == nj-1){i2d3 = i2d; i2d5 = i2d; i2d6 = i2d;}
        Real told = tt_host[i2d] + t_host[i2d];
        Real tip1 = tt_host[i2d1] + t_host[i2d1];
        Real tim1 = tt_host[i2d2] + t_host[i2d2];
        Real tjp1 = tt_host[i2d3] + t_host[i2d3];
        Real tjm1 = tt_host[i2d4] + t_host[i2d4];
        Real tip1jp1 = tt_host[i2d5] + t_host[i2d5];
        Real tim1jp1 = tt_host[i2d6] + t_host[i2d6];
        Real tip1jm1 = tt_host[i2d7] + t_host[i2d7];
        Real tim1jm1 = tt_host[i2d8] + t_host[i2d8];

        Real source = -rhs_host[i2d];
        Real residuo = (2.0f*tip1jp1+8.0f*tjp1+2.0f*tim1jp1
                     +  8.0f*tip1  -40.0f*told+8.0f*tim1
                     +  2.0f*tip1jm1+8.0f*tjm1+2.0f*tim1jm1)/(dx*dx)/12.f
                     -  source;

      if (i==0 || j==0 || i==ni-1 || j==nj-1) {residuo = 0.f;}

      rres_host[i2d] = residuo;

      x1 = dx*Real(i); x2 = dy*Real(j);
      t_analitica = -x1*x1*x2*x2*(1.f-x1*x1)*(1.f-x2*x2);
      utot = x1*x1*x2*x2*(1.f-x1*x1)*(1.f-x2*x2);
      told = t_host[i2d] + tt_host[i2d];
      umvh = told-t_analitica;
      if (umvh<0.f) umvh *= -1.f;
      errorL1 += umvh;
      errorL2 += dx*dx*umvh*umvh;
      res = res_host[i2d];
      res1 = residuo;
      resL2 += dx*dx*residuo*residuo;
      //normres += sqrt(source*source);
      normres = 2.6880683f;
      //if (normres>mnormres) mnormres = normres;
      if (res1<0.f) res1 *= -1.f;;
      if (res<0.f) res *= -1.f;
      if (res1>maxres1) maxres1 = res1;
      if (res>maxres) maxres = res;
      if (umvh>errmax) errmax = umvh;
      if (utot>maxu) maxu = utot;
    }
    //printf("Residuo  = %e \n",maxres);
    //printf("Error-L1 = %e \n",errorL1/utot);
//    printf("%e \t %e \t %e  \n",maxres/normres,maxres1/normres,errorL1/utot);
//    printf("%e \t %e \t %e \t %e \t %e  \n",maxres,maxres1,sqrt(resL2),errmax,sqrt(errorL2));
    printf("%d \t %e \t %e \t %e  \n",paso,maxres,maxres1,errmax);



    sprintf(s_pars,"MGdata-%d.vtk",paso);
    fp = fopen (s_pars, "w" );
    // Formato VTK de salida para malla estructurada
    fprintf(fp, "# vtk DataFile Version 2.0\n");
    fprintf(fp, "Sample rectilinear grid\n");
    fprintf(fp, "ASCII\n");
    fprintf(fp, "DATASET RECTILINEAR_GRID\n");
    fprintf(fp, "DIMENSIONS %d %d %d\n", ni, nj, 1);
    fprintf(fp, "X_COORDINATES %d float\n", ni);
    for (int i=0;i<ni;++i)
      fprintf(fp, "%e ", i*dx);
    fprintf(fp, "\n");
    fprintf(fp, "Y_COORDINATES %d float\n", nj);
    for (int j=0;j<nj;++j)
      fprintf(fp, "%e ", j*dy);
    fprintf(fp, "\n");
    fprintf(fp, "Z_COORDINATES %d float\n", 1);
    for (int k=0;k<1;k++)
      fprintf(fp, "%e ", 0.0f);
    fprintf(fp, "\n");
    fprintf(fp, "POINT_DATA %d\n", ni*nj);
    fprintf(fp, "SCALARS resh float\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int j = 0; j < nj; j+=1)
    for (int i = 0; i < ni; i+=1){
     int i2d = i+ni*j;
     fprintf(fp, "%e\n",res_host[i2d]);
    }
    fprintf(fp, "\n");
    fprintf(fp, "SCALARS res float\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int j = 0; j < nj; j+=1)
    for (int i = 0; i < ni; i+=1){
     int i2d = i+ni*j;
     fprintf(fp, "%e\n",rres_host[i2d]);
    }
    fprintf(fp, "\n");
    fprintf(fp, "SCALARS rhs float\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int j = 0; j < nj; j+=1)
    for (int i = 0; i < ni; i+=1){
     int i2d = i+ni*j;
     fprintf(fp, "%e\n",float(ii_host[i2d]));
    }
    fprintf(fp, "\n");
    fprintf(fp, "SCALARS rhs2 float\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int j = 0; j < nj; j+=1)
    for (int i = 0; i < ni; i+=1){
     int i2d = i+ni*j;
     fprintf(fp, "%e\n",float(ic_host[i2d]));
    }
    fprintf(fp, "\n");
    fprintf(fp, "SCALARS sol float\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int j = 0; j < nj; j+=1)
    for (int i = 0; i < ni; i+=1){
     int i2d = i+ni*j;
     fprintf(fp, "%e\n",tt_host[i2d]+t_host[i2d]);
    }

    fclose (fp);

}


//---------------
// CUDA kernels
//---------------

__global__ void inmersa_kernel (int l, int ni, int nj, int numtt, Real dx, Real dy, int *ii_data, 
                                int *ic_data, int *icn_data, Real *xc_data, Real *yc_data, Real *t_array, 
				Real *rhs_data, Real *ju_data, Real *jun_data)

{

   int i = blockIdx.x*TILE_I + threadIdx.x;
   int j = blockIdx.y*TILE_J + threadIdx.y;

   if (i<ni && j < nj) {

     // Indice del punto en la malla
     int i2d = i+j*ni; 
     int ii = ii_data[i2d]; 

     // Corregir si el punto esta cerca de la curva inmersa
     if (ii != 0) {

       // Signo del punto central
       int signo = ii;

       // Indices de los 9 vecinos 
       int i2d1 = (i+1) + j*ni;
       int i2d2 = (i-1) + j*ni;
       int i2d3 = i + (j+1)*ni;
       int i2d4 = i + (j-1)*ni;
       int i2d5 = (i+1) + (j+1)*ni;
       int i2d6 = (i-1) + (j+1)*ni;
       int i2d7 = (i+1) + (j-1)*ni;
       int i2d8 = (i-1) + (j-1)*ni;

       // Datos de vecinos
       Real tt[9];
       tt[0] = t_array[i2d];
       tt[1] = t_array[i2d1];
       tt[2] = t_array[i2d2];
       tt[3] = t_array[i2d3];
       tt[4] = t_array[i2d4];
       tt[5] = t_array[i2d5];
       tt[6] = t_array[i2d6];
       tt[7] = t_array[i2d7];
       tt[8] = t_array[i2d8];

       // Vectores radiales 
       Real drx[9];
       Real dry[9];
       drx[0] = 0.f;   dry[0] = 0.f;
       drx[1] = dx;    dry[1] = 0.f;
       drx[2] = -dx;   dry[2] = 0.f;
       drx[3] = 0.f;   dry[3] = dy;
       drx[4] = 0.f;   dry[4] = -dy;
       drx[5] = dx;    dry[5] = dy;
       drx[6] = -dx;   dry[6] = dy;
       drx[7] = dx;    dry[7] = -dy;
       drx[8] = -dx;   dry[8] = -dy;

       // Signo de los puntos cercanos a la curva
       int sv[9]; 
       sv[0] = ii_data[i2d];
       sv[1] = ii_data[i2d1]; 
       sv[2] = ii_data[i2d2];
       sv[3] = ii_data[i2d3];
       sv[4] = ii_data[i2d4];
       sv[5] = ii_data[i2d5];
       sv[6] = ii_data[i2d6]; 
       sv[7] = ii_data[i2d7]; 
       sv[8] = ii_data[i2d8];

       // ss es 0 si está del mismo lado que el nodo central,
       // 1 si el nodo es +1 o -1 si el nodo es -1
       Real ss[9]; 
       ss[0] = 0.5f*Real(signo-sv[0])*Real(abs(sv[0])); 
       ss[1] = 0.5f*Real(signo-sv[1])*Real(abs(sv[1]));
       ss[2] = 0.5f*Real(signo-sv[2])*Real(abs(sv[2]));
       ss[3] = 0.5f*Real(signo-sv[3])*Real(abs(sv[3]));
       ss[4] = 0.5f*Real(signo-sv[4])*Real(abs(sv[4]));
       ss[5] = 0.5f*Real(signo-sv[5])*Real(abs(sv[5]));
       ss[6] = 0.5f*Real(signo-sv[6])*Real(abs(sv[6]));
       ss[7] = 0.5f*Real(signo-sv[7])*Real(abs(sv[7]));
       ss[8] = 0.5f*Real(signo-sv[8])*Real(abs(sv[8]));

       Real sss[9];
       for (int k=0; k<9; k++) {
         if (signo==sv[k]) sss[k] = -signo;
       }

       int ic = ic_data[i2d];
       int icp1 = ic+1;
       int icm1 = ic-1;
       if (ic == 0) icm1 = numtt-1;
       if (ic == numtt-1) icp1 = 0;
       Real xicp1 = xc_data[icp1];
       Real yicp1 = yc_data[icp1];
       Real xicm1 = xc_data[icm1];
       Real yicm1 = yc_data[icm1];
       Real xs = (xicp1 - xicm1);
       Real ys = (yicp1 - yicm1);
       Real oor = 1.f/sqrt(xs*xs+ys*ys);
       if (oor == 0.f) oor = 1.f;
       Real alpha1 = xc_data[ic];
       Real alpha2 = yc_data[ic];

       int iin = icn_data[i2d];

       // Imponer la fuente de los saltos
       if (l==0) { 
         Real source = 0.f;
         Real ohlpq;
         Real ju = ju_data[iin];
         Real jun = jun_data[iin];
         int iii,jjj;
         Real jux = oor*ys*jun;
         Real juy = oor*xs*jun;
         for (int k=1; k<9; k++) { 
           if (ss[k] != 0) { 
             if (k==1) {iii=i+1; jjj=j; ohlpq=(8.0f/(dx*dx))/12.f;} 
             if (k==2) {iii=i-1; jjj=j; ohlpq=(8.0f/(dx*dx))/12.f;}
             if (k==3) {iii=i; jjj=j+1; ohlpq=(8.0f/(dx*dx))/12.f;}
             if (k==4) {iii=i; jjj=j-1; ohlpq=(8.0f/(dx*dx))/12.f;}
             if (k==5) {iii=i+1; jjj=j+1; ohlpq=(2.0f/(dx*dx))/12.f;}
             if (k==6) {iii=i-1; jjj=j+1; ohlpq=(2.0f/(dx*dx))/12.f;}
             if (k==7) {iii=i+1; jjj=j-1; ohlpq=(2.0f/(dx*dx))/12.f;}
             if (k==8) {iii=i-1; jjj=j-1; ohlpq=(2.0f/(dx*dx))/12.f;}
             Real xx = iii*dx; // coordenada x vecino en la malla 
             Real yy = jjj*dy; // coordenada y vecino en la malla 
             source += ss[k]*ohlpq*(ju+jux*(xx-alpha1)+juy*(yy-alpha2));

           }		  
         }

         rhs_data[i2d] -= source;

       }

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


void inmersa(int l)

{

    dim3 grid = dim3(int(float(ni)/float(TILE_I))+1, int(float(nj)/float(TILE_J))+1);
    dim3 block = dim3(TILE_I, TILE_J);

    inmersa_kernel<<<grid, block>>>(l,ni,nj,numtt,dx,dy,ii_data,ic_data,icn_data,xc_data,
		                    yc_data,t_array,rhs_data,ju_data,jun_data);

}


__global__ void relax_kernel (int ll, int ni, int nj, Real kcond, Real dt,
                              Real dx, Real dy, Real *err_data, Real *t_data,
                              Real *rhs_data, Real *res_data, Real *t_array,
                              Real *err_array, int resflag)
{
    int i, j, i2d;
    int i2d1,i2d2,i2d3,i2d4;
    int i2d5,i2d6,i2d7,i2d8;
    int i0,j0,i0ll,j0ll;
    Real told,tnow,tip1,tim1,tjp1,tjm1;
    Real tip1jp1,tim1jp1,tip1jm1,tim1jm1;
    Real enow,eip1,ejp1,eij0,eijp;
    Real dxl,dyl,dtl,source,residual;
    Real a0,a1,b0,b1;
    //Real nkgradt[9];


    i = blockIdx.x*TILE_I + threadIdx.x;
    j = blockIdx.y*TILE_J + threadIdx.y;

   if (i<ni && j < nj) {


    if (i%ll==0 && j%ll==0){

        i2d = i + j*ni;

        i2d1 = i2d + ll;
        i2d2 = i2d - ll;
        i2d3 = i2d + ll*ni;
        i2d4 = i2d - ll*ni;
        i2d5 = i2d + ll + ll*ni;
        i2d6 = i2d  - ll + ll*ni;
        i2d7 = i2d  + ll - ll*ni;
        i2d8 = i2d  - ll - ll*ni;


        if (i == 0){i2d2 = i2d;
                    i2d6 = i2d;
                    i2d8 = i2d;}
        if (j == 0){i2d4 = i2d;
                    i2d7 = i2d;
                    i2d8 = i2d;}

        if (i == ni-1){i2d1 = i2d;
                    i2d5 = i2d;
                    i2d7 = i2d;}
        if (j == nj-1){i2d3 = i2d;
                    i2d5 = i2d;
                    i2d6 = i2d;}


        dxl = Real(ll)*dx;
        dyl = Real(ll)*dy;
        dtl = dt*Real(ll*ll);

        told = t_array[i2d];
        tip1 = t_array[i2d1];
        tim1 = t_array[i2d2];
        tjp1 = t_array[i2d3];
        tjm1 = t_array[i2d4];
        tip1jp1 = t_array[i2d5];
        tim1jp1 = t_array[i2d6];
        tip1jm1 = t_array[i2d7];
        tim1jm1 = t_array[i2d8];

        source = -rhs_data[i2d];

        residual = ((2.0f*tip1jp1+8.0f*tjp1+2.0f*tim1jp1
                 +  8.0f*tip1  -40.0f*told+8.0f*tim1
                 +  2.0f*tip1jm1+8.0f*tjm1+2.0f*tim1jm1)/(dxl*dxl))/12.f
                 -  source;

//         Real sqr2 = sqrt(2.f);
   
/*         nkgradt[0] = (tim1jm1-told)/(sqr2*dxl);
         nkgradt[1] = (tjm1-told)/(dxl);
         nkgradt[2] = (tip1jm1-told)/(sqr2*dxl);
         nkgradt[3] = (tim1-told)/(dxl);
         nkgradt[4] = 0.f;
         nkgradt[5] = (tip1-told)/(dxl);
         nkgradt[6] = (tim1jp1-told)/(sqr2*dxl);
         nkgradt[7] = (tjp1-told)/(dxl);
         nkgradt[8] = (tip1jp1-told)/(sqr2*dxl);
*/
        // residual = ( nkgradt[0]+nkgradt[1]+nkgradt[2]
        //          +   nkgradt[3]+          +nkgradt[5]
        //          +   nkgradt[6]+nkgradt[7]+nkgradt[8])/(8.f*dxl)
        //          -   source;

        if (i==0 || j==0 || i==ni-1 || j==nj-1) {enow = 0.f; residual = 0.f;} else
                                                {enow = dtl*kcond*residual;}

        tnow = told + enow;
        err_data[i2d] = enow;
        if (resflag == 0) t_data[i2d]   = tnow;
        res_data[i2d] = residual;
    }
    
    else{

        i2d = i + j*ni;

        i0 = ll*int(Real(i)/Real(ll));
        j0 = ll*int(Real(j)/Real(ll));

        i0ll = i0+ll; j0ll = j0+ll;

        dxl = Real(ll)*dx;
        dyl = Real(ll)*dy;

        told = t_array[i2d];

        eij0 = err_array[i0+j0*ni];
        eip1 = err_array[i0ll+j0*ni];
        ejp1 = err_array[i0+j0ll*ni];
        eijp = err_array[i0ll+j0ll*ni];

        a1 = (Real(i-i0)*dx)/dxl;
        a0 = 1.0f-a1;
        b1 = (Real(j-j0)*dy)/dyl;
        b0 = 1.0f-b1;

        enow = a0*b0*eij0 + a1*b0*eip1 + a0*b1*ejp1 + a1*b1*eijp;

        tnow = told + enow;

        t_data[i2d] = tnow;

    }
	
  }

}
					

void relax(int ll, Real dx, Real dy, Real kcond, Real dt, int ni, int nj, Real *t_data, 
           Real *err_data, Real *t_array, Real *err_array, Real *rhs_data, Real *res_data, int resflag)
{
  
  
    cudaMemcpy(t_array, t_data, sizeof(Real)*ni*nj, cudaMemcpyDeviceToDevice);
    cudaMemcpy(err_array, err_data, sizeof(Real)*ni*nj, cudaMemcpyDeviceToDevice);									   

    dim3 grid = dim3(int(float(ni)/float(TILE_I))+1, int(float(nj)/float(TILE_J))+1);
    dim3 block = dim3(TILE_I, TILE_J);
    
    relax_kernel<<<grid, block>>>(ll,ni,nj,kcond,dt,dx,dy, 
                                         err_data,t_data,rhs_data,res_data,t_array,err_array,resflag);
    
}


//-------------------------//
//   Restriction operator  //	
//-------------------------//
__global__ void restrict_kernel (int ll, int ni, int nj, Real *rhsr_data, Real *rhs_array)

{
	int i, j, i2d;
	int i2d1,i2d2,i2d3,i2d4,i2d5,i2d6,i2d7,i2d8;
	int imll,jmll,ipll,jpll;
	Real rtot;
	Real rij,rim1jm1,rim1j,rim1jp1,rijm1,rijp1,rip1jm1,rip1j,rip1jp1;
	//Real atot;
	//Real aij,aim1jm1,aim1j,aim1jp1,aijm1,aijp1,aip1jm1,aip1j,aip1jp1;
	
	i = blockIdx.x*TILE_I + threadIdx.x;
	j = blockIdx.y*TILE_J + threadIdx.y;

   if (i<ni && j < nj) {
	   
	i2d = i + j*ni;
	
	imll = i-ll;
	jmll = j-ll;
	ipll = i+ll;
	jpll = j+ll;
	
	if (imll < 0) imll = 0;
	if (jmll < 0) jmll = 0;
	if (ipll > ni-1) ipll = ni-1;
	if (jpll > nj-1) jpll = nj-1;
	
	i2d1 = (imll)+ni*(jmll);
	i2d2 = (imll)+ni*(j);
	i2d3 = (imll)+ni*(jpll);
	i2d4 = (i)   +ni*(jmll);
	i2d5 = (i)   +ni*(jpll);
	i2d6 = (ipll)+ni*(jmll);
	i2d7 = (ipll)+ni*(j);
	i2d8 = (ipll)+ni*(jpll); 
		  
	rij     = rhs_array[i2d];
	rim1jm1 = rhs_array[i2d1];
	rim1j   = rhs_array[i2d2];
	rim1jp1 = rhs_array[i2d3];
	rijm1   = rhs_array[i2d4];
	rijp1   = rhs_array[i2d5];
	rip1jm1 = rhs_array[i2d6];
	rip1j   = rhs_array[i2d7];
	rip1jp1 = rhs_array[i2d8];

//	aij     = 1.0f; 
//	aim1jm1 = 1.0f;
//	aim1j   = 1.0f;
//	aim1jp1 = 1.0f;
//	aijm1   = 1.0f;
//	aijp1   = 1.0f;
//	aip1jm1 = 1.0f;
//	aip1j   = 1.0f;
//	aip1jp1 = 1.0f;
	
//	if (i<ll) {  
//	rim1jm1 = 0.0f; rim1jp1 = 0.0f; rim1j = 0.0f;
//	aim1jm1 = 0.0f; aim1jp1 = 0.0f; aim1j = 0.0f;
//	}
//	if (i>ni-1-ll) {
//	rip1jm1 = 0.0f; rip1j = 0.0f; rip1jp1 = 0.0f;
//	aip1jm1 = 0.0f; aip1j = 0.0f; aip1jp1 = 0.0f;
//	}
//	if (j<ll) {
//	rip1jm1 = 0.0f; rijm1 = 0.0f; rim1jm1 = 0.0f;
//	aip1jm1 = 0.0f; aijm1 = 0.0f; aim1jm1 = 0.0f;
//	}
//	if (j>nj-1-ll) {
//	rim1jp1 = 0.0f; rijp1 = 0.0f; rip1jp1 = 0.0f;
//	aim1jp1 = 0.0f; aijp1 = 0.0f; aip1jp1 = 0.0f;
//	}
	
	rtot = rij + 0.25f*(rim1jm1+rim1jp1+rip1jm1+rip1jp1) + 0.5f*(rim1j+rijm1+rijp1+rip1j);
//	atot = aij + 0.25f*(aim1jm1+aim1jp1+aip1jm1+aip1jp1) + 0.5f*(aim1j+aijm1+aijp1+aip1j);

//	rhsr_data[i2d] = rtot/atot;

    rhsr_data[i2d] = rtot/4.f;

  }

}

void restrict(int ll, int ni, int nj, Real *rhs_array, Real *rhsr_data)
{
    cudaMemcpy(rhs_array, rhsr_data, sizeof(Real)*ni*nj, cudaMemcpyDeviceToDevice);
									   
    dim3 grid = dim3(int(float(ni)/float(TILE_I))+1, int(float(nj)/float(TILE_J))+1);
    dim3 block = dim3(TILE_I, TILE_J);

    restrict_kernel<<<grid, block>>>(ll,ni,nj,rhsr_data,rhs_array);

}
 	
__global__ void suma_kernel (int ni, int nj, Real *t_data, Real *tt_data)

{
    int i, j, i2d;
  
    i = blockIdx.x*TILE_I + threadIdx.x;
    j = blockIdx.y*TILE_J + threadIdx.y;

    i2d = i + j*ni;

   if (i<ni && j < nj) {
    tt_data[i2d] += t_data[i2d]; 
    t_data[i2d] = 0.f;
   }
}


void suma(int ni, int nj, Real *t_data, Real *tt_data)

{
    dim3 grid = dim3(int(float(ni)/float(TILE_I))+1, int(float(nj)/float(TILE_J))+1);
    dim3 block = dim3(TILE_I, TILE_J);

    suma_kernel<<<grid, block>>>(ni, nj, t_data, tt_data);

}


/////////////////////////////////////////////////////////////////////////////////////
/// Cond de Frontera
////////////////////////////////////////////////////////////////////////////////////


__global__ void apply_BCs_kernel (int ni, int nj, Real *t_data, Real *err_data)

{
    int i, j, i2d;
   
    i = blockIdx.x*TILE_I + threadIdx.x;
    j = blockIdx.y*TILE_J + threadIdx.y;

    i2d = i + j*ni;
 
   if (i<ni && j < nj) {

    if (i == 0 || i==ni-1 || j == 0 || j==nj-1) { t_data[i2d] = 0.f;}
    if (i == 0 || i==ni-1 || j == 0 || j==nj-1) { err_data[i2d] = 0.f;}
   }
}


void apply_BCs(int ni, int nj, Real *t_data, Real *err_data)

{
    dim3 grid = dim3(int(float(ni)/float(TILE_I))+1, int(float(nj)/float(TILE_J))+1);
    dim3 block = dim3(TILE_I, TILE_J);

    apply_BCs_kernel<<<grid, block>>>(ni, nj, t_data, err_data);
    
}

