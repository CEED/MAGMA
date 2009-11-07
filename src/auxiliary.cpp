/*
    -- MAGMA (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       June 2009
*/

#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cublas.h"
#include <stdio.h>
#include "magma.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Get current time
*/ 
TimeStruct get_current_time(void)
{
  static struct timeval  time_val;
  static struct timezone time_zone;

  TimeStruct time;

  cudaThreadSynchronize();
  gettimeofday(&time_val, &time_zone);

  time.sec  = time_val.tv_sec;
  time.usec = time_val.tv_usec;
  return (time);
}


/* ////////////////////////////////////////////////////////////////////////////
   -- End elapsed time
*/ 
double GetTimerValue(TimeStruct time_1, TimeStruct time_2)
{
  int sec, usec;

  sec  = time_2.sec  - time_1.sec;
  usec = time_2.usec - time_1.usec;

  return (1000.*(double)(sec) + (double)(usec) * 0.001);
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Print the available GPU devices
*/
void printout_devices( )
{
  int ndevices;
  cuDeviceGetCount( &ndevices );
  for( int idevice = 0; idevice < ndevices; idevice++ )
    {
      char name[200];
      unsigned int totalMem, clock;
      CUdevice dev;

      cuDeviceGet( &dev, idevice );
      cuDeviceGetName( name, sizeof(name), dev );
      cuDeviceTotalMem( &totalMem, dev );
      cuDeviceGetAttribute( (int*)&clock,
                            CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev );
      printf( "device %d: %s, %.1f MHz clock, %.1f MB memory\n",
              idevice, name, clock/1000.f, totalMem/1024.f/1024.f );
    }
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Put 0s in the upper triangular part of a panel (and 1s on the diagonal)
      if uplo is 'U'/'u', or 0s in the lower triangular part of a panel (and 
      1s on the diagonal) if uplo is 'L'/'l'.
      This is auxiliary function used in geqrf and geqlf.  
*/
void spanel_to_q(char uplo, int ib, float *a, int lda, float *work){
  int i, j, k = 0;
  float *col;

  if (uplo == 'U' || uplo == 'u'){
    for(i=0; i<ib; i++){
      col = a + i*lda;
      for(j=0; j<i; j++){
	work[k++] = col[j];
	col[j] = 0.;
      }
      work[k++] = col[i];
      col[j] = 1.;
    }
  }
  else {
    for(i=0; i<ib; i++){
      col = a + i*lda;
      work[k++] = col[i];
      col[i] = 1.;
      for(j=i+1; j<ib; j++){
        work[k++] = col[j];
        col[j] = 0.;
      }
    }
  }
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Restores a panel (after call to "panel_to_q").
      This isauxiliary function usedin geqrf and geqlf.
*/
void sq_to_panel(char uplo, int ib, float *a, int lda, float *work){
  int i, j, k = 0;
  float *col;

  if (uplo == 'U' || uplo == 'u'){
    for(i=0; i<ib; i++){
      col = a + i*lda;
      for(j=0; j<=i; j++)
	col[j] = work[k++];
    }
  }
  else {
    for(i=0; i<ib; i++){
      col = a + i*lda;
      for(j=i; j<ib; j++)
        col[j] = work[k++];
    }
  }
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Put 0s in the upper triangular part of a panel (and 1s on the diagonal)
*/
void cpanel_to_q(char uplo, int ib, float2 *a, int lda, float2 *work){
  int i, j, k = 0;
  float2 *col;
  for(i=0; i<ib; i++){
    col = a + i*lda;
    for(j=0; j<i; j++){
      work[k  ].x = col[j].x;
      work[k++].y = col[j].y;
      col[j].x = col[j].y = 0.;
    }
    work[k  ].x = col[i].x;
    work[k++].y = col[i].y;
    col[j].x = 1.;
    col[j].y = 0.;
  }
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Restores a panel (after call to "panel_to_q")
*/
void cq_to_panel(char uplo, int ib, float2 *a, int lda, float2 *work){
  int i, j, k = 0;
  float2 *col;
  for(i=0; i<ib; i++){
    col = a + i*lda;
    for(j=0; j<=i; j++){
      col[j].x = work[k  ].x;
      col[j].y = work[k++].y;
    }
  }
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Put 0s in the upper triangular part of a panel (and 1s on the diagonal)
*/
void dpanel_to_q(int ib, double *a, int lda, double *work){
  int i, j, k = 0;
  double *col;
  for(i=0; i<ib; i++){
    col = a + i*lda;
    for(j=0; j<i; j++){
      work[k++] = col[j];
      col[j] = 0.;
    }
    work[k++] = col[i];
    col[j] = 1.;
  }
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Restores a panel (after call to "panel_to_q")
*/
void dq_to_panel(int ib, double *a, int lda, double *work){
  int i, j, k = 0;
  double *col;
  for(i=0; i<ib; i++){
    col = a + i*lda;
    for(j=0; j<=i; j++)
      col[j] = work[k++];
  }
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Auxiliary function: ipiv(i) indicates that row i has been swapped with 
      ipiv(i) from top to bottom. This function rearranges ipiv into newipiv
      where row i has to be moved to newipiv(i). The new pivoting allows for
      parallel processing vs the original one assumes a specific ordering and
      has to be done sequentially.
*/
void swp2pswp(int n, int *ipiv, int *newipiv){
  int i, newind, ind;
  
  for(i=0; i<n; i++)
    newipiv[i] = -1;
  
  for(i=0; i<n; i++){
    newind = ipiv[i] - 1;
    if (newipiv[newind] == -1) {
      if (newipiv[i]==-1){
        newipiv[i] = newind;
        if (newind>i)
          newipiv[newind]= i;
      }
      else
        {
          ind = newipiv[i];
          newipiv[i] = newind;
          if (newind>i)
            newipiv[newind]= ind;
        }
    }
    else {
      if (newipiv[i]==-1){
        if (newind>i){
          ind = newipiv[newind];
          newipiv[newind] = i;
          newipiv[i] = ind;
        }
        else
          newipiv[i] = newipiv[newind];
      }
      else{
	ind = newipiv[i];
	newipiv[i] = newipiv[newind];
        if (newind > i)
          newipiv[newind] = ind;
      }
    }
  }
}
