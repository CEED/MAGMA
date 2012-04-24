/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011
*/

#include "common_magma.h"
#include <assert.h>

/* ////////////////////////////////////////////////////////////////////////////
   -- Get number of GPUs to use from $MAGMA_NUM_GPUS environment variable.
   @author Mark Gates
*/
extern "C"
int magma_num_gpus( void )
{
    const char *ngpu_str = getenv("MAGMA_NUM_GPUS");
    int ngpu = 1;
    if ( ngpu_str != NULL ) {
        char* endptr;
        ngpu = strtol( ngpu_str, &endptr, 10 );
        int ndevices;
        cudaGetDeviceCount( &ndevices );
        // if *endptr == '\0' then entire string was valid number (or empty)
        if ( ngpu < 1 or *endptr != '\0' ) {
            ngpu = 1;
            fprintf( stderr, "$MAGMA_NUM_GPUS=%s is an invalid number; using %d GPU.\n",
                     ngpu_str, ngpu );
        }
        else if ( ngpu > MagmaMaxGPUs or ngpu > ndevices ) {
            ngpu = min( ndevices, MagmaMaxGPUs );
            fprintf( stderr, "$MAGMA_NUM_GPUS=%s exceeds MagmaMaxGPUs=%d or available GPUs=%d; using %d GPUs.\n",
                     ngpu_str, MagmaMaxGPUs, ndevices, ngpu );
        }
        assert( 1 <= ngpu and ngpu <= ndevices );
    }
    return ngpu;
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Print the available GPU devices
   @author Mark Gates
*/
extern "C"
void printout_devices( )
{
    int ndevices;
    cudaGetDeviceCount( &ndevices );
    for( int idevice = 0; idevice < ndevices; idevice++ ) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties( &prop, idevice );
        printf( "device %d: %s, %.1f MHz clock, %.1f MB memory, capability %d.%d\n",
                idevice,
                prop.name,
                prop.clockRate / 1000.,
                prop.totalGlobalMem / (1024.*1024.),
                prop.major,
                prop.minor );
    }
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Auxiliary function: ipiv(i) indicates that row i has been swapped with 
      ipiv(i) from top to bottom. This function rearranges ipiv into newipiv
      where row i has to be moved to newipiv(i). The new pivoting allows for
      parallel processing vs the original one assumes a specific ordering and
      has to be done sequentially.
*/
extern "C"
void swp2pswp(char trans, int n, int *ipiv, int *newipiv){
  int i, newind, ind;
  char            trans_[2] = {trans, 0};
  long int    notran = lapackf77_lsame(trans_, "N");

  for(i=0; i<n; i++)
    newipiv[i] = -1;
  
  if (notran){
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
  } else {
    for(i=n-1; i>=0; i--){
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
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Auxiliary function: used for debugging. Given a pointer to floating
      point number on the GPU memory, the function returns the value
      at that location.
*/
extern "C"
float getv(float *da){
  float res[1];
  cublasGetVector(1, sizeof(float), da, 1, res, 1);
  return res[0];
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Auxiliary function sp_cat
*/
extern "C"
int sp_cat(char *lp, char *rpp[], magma_int_t *rnp, magma_int_t*np, magma_int_t ll)
{
  magma_int_t i, n, nc;
  char *f__rp;

  n = (int)*np;
  for(i = 0 ; i < n ; ++i)
    {
      nc = ll;
      if(rnp[i] < nc)
        nc = rnp[i];
      ll -= nc;
      f__rp = rpp[i];
      while(--nc >= 0)
        *lp++ = *f__rp++;
    }
  while(--ll >= 0)
    *lp++ = ' ';

  return 0;
}
