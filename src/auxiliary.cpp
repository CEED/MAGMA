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
   -- LAPACK auxiliary function sqrt02
*/
int sqrt02(int *m, int *n, int *k, float *a, float *af, float *q,
           float *r__, int *lda, float *tau, float *work,
           int *lwork, float *rwork, float *result){
  int a_dim1, a_offset, af_dim1, af_offset, q_dim1, q_offset, r_dim1, 
    r_offset, i__1;

#define max(a,b)       (((a)>(b))?(a):(b))

  static float c_b4 = -1e10f;
  static float c_b9 = 0.f;
  static float c_b14 = -1.f;
  static float c_b15 = 1.f;

  static float eps;
  static int info;
  static float resid;

  r_dim1 = *lda;
  r_offset = 1 + r_dim1;
  r__ -= r_offset;
  q_dim1 = *lda;
  q_offset = 1 + q_dim1;
  q -= q_offset;
  af_dim1 = *lda;
  af_offset = 1 + af_dim1;
  af -= af_offset;
  a_dim1 = *lda;
  a_offset = 1 + a_dim1;
  a -= a_offset;
  --tau;
  --work;
  --rwork;
  --result;

  /* Function Body */
  eps = slamch_("Epsilon");
  
  /*     Copy the first k columns of the factorization to the array Q */
  slaset_("Full", m, n, &c_b4, &c_b4, &q[q_offset], lda);
  i__1 = *m - 1;
  slacpy_("Lower", &i__1, k, &af[af_dim1 + 2], lda, &q[q_dim1 + 2], lda);

  /*     Generate the first n columns of the matrix Q */
  sorgqr_(m, n, k, &q[q_offset], lda, &tau[1], &work[1], lwork, &info);

  /*     Copy R(1:n,1:k) */
  slaset_("Full", n, k, &c_b9, &c_b9, &r__[r_offset], lda);
  slacpy_("Upper", n, k, &af[af_offset], lda, &r__[r_offset], lda);

  /*     Compute R(1:n,1:k) - Q(1:m,1:n)' * A(1:m,1:k) */
  sgemm_("Transpose", "No transpose", n, k, m, &c_b14, &q[q_offset], lda, &
	 a[a_offset], lda, &c_b15, &r__[r_offset], lda);

  /*     Compute norm( R - Q'*A ) / ( M * norm(A) * EPS ) . */
  double anorm = slange_("1", m, k, &a[a_offset], lda, &rwork[1]);
  printf("norm of a = %f\n",anorm);
  resid = slange_("1", n, k, &r__[r_offset], lda, &rwork[1]);
  printf("norm of r = %f\n",resid);
  if (anorm > 0.f) {
    result[1] = resid / (float) max(1,*m) / anorm / eps;
  } else {
    result[1] = 0.f;
  }

  /*     Compute I - Q'*Q */
  slaset_("Full", n, n, &c_b9, &c_b15, &r__[r_offset], lda);
  ssyrk_("Upper", "Transpose", n, m, &c_b14, &q[q_offset], lda, &c_b15, &
	 r__[r_offset], lda);

  /*     Compute norm( I - Q'*Q ) / ( M * EPS ) . */
  resid = slansy_("1", "Upper", n, &r__[r_offset], lda, &rwork[1]);
  result[2] = resid / (float) max(1,*m) / eps;

  return 0;

#undef max
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Put 0s in the upper triangular part of a panel (and 1s on the diagonal)
*/
void cpanel_to_q(int ib, float2 *a, int lda, float2 *work){
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
void cq_to_panel(int ib, float2 *a, int lda, float2 *work){
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
   -- LAPACK auxiliary function dqrt02
*/
int dqrt02(int *m, int *n, int *k, double *a, double *af, double *q,
           double *r__, int *lda, double *tau, double *work,
           int *lwork, double *rwork, double *result){
  int a_dim1, a_offset, af_dim1, af_offset, q_dim1, q_offset, r_dim1, 
    r_offset, i__1;

#define max(a,b)       (((a)>(b))?(a):(b))

  static double c_b4 = -1e10f;
  static double c_b9 = 0.f;
  static double c_b14 = -1.f;
  static double c_b15 = 1.f;

  static double eps;
  static int info;
  static double resid;

  r_dim1 = *lda;
  r_offset = 1 + r_dim1;
  r__ -= r_offset;
  q_dim1 = *lda;
  q_offset = 1 + q_dim1;
  q -= q_offset;
  af_dim1 = *lda;
  af_offset = 1 + af_dim1;
  af -= af_offset;
  a_dim1 = *lda;
  a_offset = 1 + a_dim1;
  a -= a_offset;
  --tau;
  --work;
  --rwork;
  --result;

  /* Function Body */
  eps = dlamch_("Epsilon");
  
  /*     Copy the first k columns of the factorization to the array Q */
  dlaset_("Full", m, n, &c_b4, &c_b4, &q[q_offset], lda);
  i__1 = *m - 1;
  dlacpy_("Lower", &i__1, k, &af[af_dim1 + 2], lda, &q[q_dim1 + 2], lda);

  /*     Generate the first n columns of the matrix Q */
  dorgqr_(m, n, k, &q[q_offset], lda, &tau[1], &work[1], lwork, &info);

  /*     Copy R(1:n,1:k) */
  dlaset_("Full", n, k, &c_b9, &c_b9, &r__[r_offset], lda);
  dlacpy_("Upper", n, k, &af[af_offset], lda, &r__[r_offset], lda);

  /*     Compute R(1:n,1:k) - Q(1:m,1:n)' * A(1:m,1:k) */
  dgemm_("Transpose", "No transpose", n, k, m, &c_b14, &q[q_offset], lda, &
	 a[a_offset], lda, &c_b15, &r__[r_offset], lda);

  /*     Compute norm( R - Q'*A ) / ( M * norm(A) * EPS ) . */
  double anorm = dlange_("1", m, k, &a[a_offset], lda, &rwork[1]);
  printf("norm of a = %f\n",anorm);
  resid = dlange_("1", n, k, &r__[r_offset], lda, &rwork[1]);
  printf("norm of r = %f\n",resid);
  if (anorm > 0.f) {
    result[1] = resid / (double) max(1,*m) / anorm / eps;
  } else {
    result[1] = 0.f;
  }

  /*     Compute I - Q'*Q */
  dlaset_("Full", n, n, &c_b9, &c_b15, &r__[r_offset], lda);
  dsyrk_("Upper", "Transpose", n, m, &c_b14, &q[q_offset], lda, &c_b15, &
	 r__[r_offset], lda);

  /*     Compute norm( I - Q'*Q ) / ( M * EPS ) . */
  resid = dlansy_("1", "Upper", n, &r__[r_offset], lda, &rwork[1]);
  result[2] = resid / (double) max(1,*m) / eps;

  return 0;

#undef max
}
