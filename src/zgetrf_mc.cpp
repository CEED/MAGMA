/*
    -- MAGMA (version 0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2009

       @precisions normal z -> s d c

*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cublas.h>
#include <magma.h>
#include <quark.h>
//#include <cblas.h>

// trace
#include "event_trace.h"

/*

int    event_num        [MAX_THREADS];
double event_start_time [MAX_THREADS];
double event_end_time   [MAX_THREADS];
double event_log        [MAX_THREADS][MAX_EVENTS];
char *event_label       [MAX_THREADS][MAX_EVENTS];
int log_events = 1;
char tasklabel[200];
char taskcolor[200];

double get_current_timee(void) {
  struct timeval tp;
  gettimeofday( &tp, NULL );
  return tp.tv_sec + 1e-6 * tp.tv_usec;
}

// dump trace
void dump_trace(int cores_num) {
  char trace_file_name[32];
  FILE *trace_file;
  double end;
  int event, core;
  double scale = 30000.0;

  sprintf(trace_file_name, "trace.svg");
  trace_file = fopen(trace_file_name, "w");

  int x_end_max = -1;

  for (core = 0; core < cores_num; core++) {
    event = event_num[core]-4;
    end = event_log[core][event+2] - event_log[0][1];
    x_end_max = max(x_end_max, (int)(end * scale) + 20);
  }

fprintf(trace_file,
"<?xml version=\"1.0\" standalone=\"no\"?>"
"<svg version=\"1.1\" baseProfile=\"full\" xmlns=\"http://www.w3.org/2000/svg\" "
"xmlns:xlink=\"http://www.w3.org/1999/xlink\" xmlns:ev=\"http://www.w3.org/2001/xml-events\"  "
">\n"
"  <g font-size=\"20\">\n");

for (core = 0; core < cores_num; core++)

for (event = 0; event < event_num[core]; event += 4) {

int    tag   = (int)event_log[core][event+0];
double start =      event_log[core][event+1];
double end   =      event_log[core][event+2];
int    color = (int)event_log[core][event+3];
char   *label = event_label[core][event+0];

start -= event_log[0][1];
end   -= event_log[0][1];

fprintf(trace_file,
"    "
"<rect x=\"%.2lf\" y=\"%.0lf\" width=\"%.2lf\" height=\"%.0lf\" "
"fill=\"#%06x\" stroke=\"#000000\" stroke-width=\"1\"/>\n",
start * scale,  // x
core * 100.0,   // y
(end - start) * scale, // width
90.0,           // height
color);

  fprintf(trace_file,
  "    "
  "<text x=\"%.2lf\" y=\"%.0lf\" font-size=\"20\" fill=\"black\">"
  "%s"
  "</text>\n",
  start * scale + 10, // x
  core * 100.0 + 20, // y
  label);

}
//}

fprintf(trace_file,
"  </g>\n"
"</svg>\n");

fclose(trace_file);
}

*/


#define  A(m,n) (a+(n)*(*lda)+(m))

extern "C" int EN_BEE;

extern "C" int TRACE;

void SCHED_sgemm(Quark* quark)
{
int M;
int N;
int K;
cuDoubleComplex *A1;
int LDA;
cuDoubleComplex *A2;
cuDoubleComplex *A3;

cuDoubleComplex mone = MAGMA_Z_NEG_ONE;
cuDoubleComplex one = MAGMA_Z_ONE;
	
int color;

quark_unpack_args_7(quark, M, N, K, A1, LDA, A2, A3);

//if (TRACE == 1)
  //core_event_start(QUARK_Thread_Rank(quark));

blasf77_zgemm("n", "n", 
  &M, &N, &K, &mone, A1, &LDA, A2, &LDA, &one, A3, &LDA);

//if (TRACE == 1) {
  //core_event_end(QUARK_Thread_Rank(quark));
  //core_log_event_label(0x0000ff,QUARK_Thread_Rank(quark),0,"");
//}

}

void SCHED_panel_update(Quark* quark)
{
int N;
cuDoubleComplex *A1;
int LDA;
int K2;
int *IPIV;
cuDoubleComplex *A2;
int M;
int K;
cuDoubleComplex *A3;
cuDoubleComplex *A4;

int ione=1;
cuDoubleComplex mone = MAGMA_Z_NEG_ONE;
cuDoubleComplex one = MAGMA_Z_ONE;
	
int color;

quark_unpack_args_10(quark, N, A1, LDA, K2, IPIV, A2, M, K, A3, A4);

//if (TRACE == 1)
  //core_event_start(QUARK_Thread_Rank(quark));

lapackf77_zlaswp(&N, A1, &LDA, &ione, &K2, IPIV, &ione); 

blasf77_ztrsm("l", "l", "n", "u",
  &K2, &N, &one, A2, &LDA, A1, &LDA);

if (M > 0) {

blasf77_zgemm("n","n", 
  &M, &N, &K, &mone, A3, &LDA, A1, &LDA, &one, A4, &LDA);

}

//if (TRACE == 1) {
  //core_event_end(QUARK_Thread_Rank(quark));
  //core_log_event_label(0x00ffff,QUARK_Thread_Rank(quark),0,"");
//}

}

void SCHED_sgetrf(Quark* quark)
{
int M;
int N;
cuDoubleComplex *A;
int LDA;
int *IPIV;

int *iinfo;

int info;

quark_unpack_args_5(quark, M, N, A, LDA, IPIV);

//if (TRACE == 1)
  //core_event_start(QUARK_Thread_Rank(quark));

lapackf77_zgetrf(&M, &N, A, &LDA, IPIV, &info); 

if (info > 0) {
  iinfo[1] = iinfo[0] + info;
}

//if (TRACE == 1) {
  //core_event_end(QUARK_Thread_Rank(quark));
  //core_log_event_label(0x00ff00,QUARK_Thread_Rank(quark),0,"");
//}

}

void SCHED_slaswp(Quark* quark)
{
int N;
cuDoubleComplex *A;
int LDA;
int K2;
int *IPIV;

int ione=1;

quark_unpack_args_5(quark, N, A, LDA, K2, IPIV);

//if (TRACE == 1)
  //core_event_start(QUARK_Thread_Rank(quark));

lapackf77_zlaswp(&N, A, &LDA, &ione, &K2, IPIV, &ione); 

//if (TRACE == 1) {
  //core_event_end(QUARK_Thread_Rank(quark));
  //core_log_event_label(0x9932cc,QUARK_Thread_Rank(quark),0,"");
//}

}

extern "C" int 
magma_zgetrf_mc(
int *m,
int *n,
cuDoubleComplex *a,
int *lda,
int *ipiv,
int *info)
{
/*  -- MAGMA (version 0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2009

    Purpose   
    =======   

    SGETRF computes an LU factorization of a general M-by-N matrix A   
    using partial pivoting with row interchanges.   

    The factorization has the form   
       A = P * L * U   
    where P is a permutation matrix, L is lower triangular with unit   
    diagonal elements (lower trapezoidal if m > n), and U is upper   
    triangular (upper trapezoidal if m < n).   

    This is the right-looking Level 3 BLAS version of the algorithm.   

    Arguments   
    =========   

    M       (input) INTEGER   
            The number of rows of the matrix A.  M >= 0.   

    N       (input) INTEGER   
            The number of columns of the matrix A.  N >= 0.   

    A       (input/output) REAL array, dimension (LDA,N)   
            On entry, the M-by-N matrix to be factored.   
            On exit, the factors L and U from the factorization   
            A = P*L*U; the unit diagonal elements of L are not stored.   

    LDA     (input) INTEGER   
            The leading dimension of the array A.  LDA >= max(1,M).   

    IPIV    (output) INTEGER array, dimension (min(M,N))   
            The pivot indices; for 1 <= i <= min(M,N), row i of the   
            matrix was interchanged with row IPIV(i).   

    INFO    (output) INTEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument had an illegal value   
            > 0:  if INFO = i, U(i,i) is exactly zero. The factorization   
                  has been completed, but the factor U is exactly   
                  singular, and division by zero will occur if it is used   
                  to solve a system of equations.   

    =====================================================================    */

  int i,j,l;

  int ii,jj,ll;

  void *fakedep;

  int ione=1;

  cuDoubleComplex fone = MAGMA_Z_ONE;
  cuDoubleComplex mone = MAGMA_Z_NEG_ONE;

  int M,N,MM,NN,MMM,K;

  int priority=0;

  *info = 0;

  int nb = EN_BEE;

  // check arguments
  if (*m < 0) {
    *info = -1;
  } else if (*n < 0) {
    *info = -2;
  } else if (*lda < max(1,*m)) {
    *info = -4;
  }
  if (*info != 0)
    return 0;

  int k = min(*m,*n);

  int iinfo[2];
    iinfo[1] = 0;

  char label[10000];

  // start scheduler 
  Quark *quark = QUARK_New(4);

  ii = -1;

  // loop across diagonal blocks
  for (i = 0; i < k; i += nb) {

    ii++;

    jj = -1;

    priority = 10000 - ii;

    // update panels in left looking fashion
    for (j = 0; j < i; j += nb) { 

      jj++;

      NN=min(nb,(*n)-i);
      MM=min(nb,(*m)-j);

      l = j + nb;

      MMM = min(nb,(*m)-l);

      sprintf(label, "UPDATE %d %d", ii, jj);
  
      QUARK_Insert_Task(quark, SCHED_panel_update, 0,
        sizeof(int),             &NN,      VALUE,
        sizeof(cuDoubleComplex)*(*m)*(*n), A(j,i),   INOUT,
        sizeof(int),             m,        VALUE,
        sizeof(int),             &MM,      VALUE,
        sizeof(cuDoubleComplex)*nb,        &ipiv[j], INPUT,
        sizeof(cuDoubleComplex)*(*m)*(*n), A(j,j),   INPUT,
        sizeof(int),             &MMM,     VALUE,
        sizeof(int),             &nb,      VALUE,
        sizeof(cuDoubleComplex)*(*m)*(*n), A(l,j),   INPUT,
        sizeof(cuDoubleComplex)*(*m)*(*n), A(l,i),   INOUT,
        sizeof(int),             &priority,VALUE | TASK_PRIORITY,
        sizeof(cuDoubleComplex)*(*m)*(*n), A(i,i),   OUTPUT,
        strlen(label)+1,         label,    VALUE | TASKLABEL,
        5,                       "cyan",   VALUE | TASKCOLOR,
        0);

      ll = jj + 1;

      // split gemm into tiles
      for (l = j + (2*nb); l < (*m); l += nb) {

        ll++;

        MMM = min(nb,(*m)-l);

        fakedep = (void *)(intptr_t)(j+1);

        sprintf(label, "GEMM %d %d %d", ii, jj, ll);
  
        QUARK_Insert_Task(quark, SCHED_sgemm, 0,
          sizeof(int),             &MMM,     VALUE,
          sizeof(int),             &NN,      VALUE,
          sizeof(int),             &nb,      VALUE,
          sizeof(cuDoubleComplex)*(*m)*(*n), A(l,j),   INPUT,
          sizeof(int),             m,        VALUE,
          sizeof(cuDoubleComplex)*(*m)*(*n), A(j,i),   INPUT,
          sizeof(cuDoubleComplex)*(*m)*(*n), A(l,i),   INOUT,
          sizeof(int),             &priority,VALUE | TASK_PRIORITY,
          sizeof(cuDoubleComplex)*(*m)*(*n), A(i,i),   OUTPUT | GATHERV,
          sizeof(void*),           fakedep,  OUTPUT | GATHERV,
          strlen(label)+1,         label,    VALUE | TASKLABEL,
          5,                       "blue",   VALUE | TASKCOLOR,
          0);

      }  

    }

    M=(*m)-i;
    N=min(nb,(*n)-i);

    iinfo[0] = i;

    sprintf(label, "GETRF %d", ii);
  
    QUARK_Insert_Task(quark, SCHED_sgetrf, 0,
      sizeof(int),             &M,       VALUE,
      sizeof(int),             &N,       VALUE,
      sizeof(cuDoubleComplex)*(*m)*(*n), A(i,i),   INOUT,
      sizeof(int),             m,        VALUE,
      sizeof(cuDoubleComplex)*nb,        &ipiv[i], OUTPUT,
      sizeof(int),             iinfo,    OUTPUT,
      sizeof(int),             &priority,VALUE | TASK_PRIORITY,
      strlen(label)+1,         label,    VALUE | TASKLABEL,
      6,                       "green",  VALUE | TASKCOLOR,
      0);

  }

  K = (*m)/nb;

  if ((K*nb)==(*m)) {
    ii = K - 1;
    K = *m;
  } else {
    ii = k;
    K = (K+1)*nb;
  }

  priority = 0;

  // if n > m
  for (i = K; i < (*n); i += nb) {

    ii++;

    jj = -1;

    // update remaining panels in left looking fashion
    for (j = 0; j < (*m); j += nb) { 

      jj++;

      NN=min(nb,(*n)-i);
      MM=min(nb,(*m)-j);

      l = j + nb;

      MMM = min(nb,(*m)-l);

      sprintf(label, "UPDATE %d %d", ii, jj);
  
      QUARK_Insert_Task(quark, SCHED_panel_update, 0,
        sizeof(int),             &NN,      VALUE,
        sizeof(cuDoubleComplex)*(*m)*(*n), A(j,i),   INOUT,
        sizeof(int),             m,        VALUE,
        sizeof(int),             &MM,      VALUE,
        sizeof(cuDoubleComplex)*nb,        &ipiv[j], INPUT,
        sizeof(cuDoubleComplex)*(*m)*(*n), A(j,j),   INPUT,
        sizeof(int),             &MMM,     VALUE,
        sizeof(int),             &nb,      VALUE,
        sizeof(cuDoubleComplex)*(*m)*(*n), A(l,j),   INPUT,
        sizeof(cuDoubleComplex)*(*m)*(*n), A(l,i),   INOUT,
        sizeof(int),             &priority,VALUE | TASK_PRIORITY,
        sizeof(cuDoubleComplex)*(*m)*(*n), A(i,i),   OUTPUT,
        strlen(label)+1,         label,    VALUE | TASKLABEL,
        5,                       "cyan",   VALUE | TASKCOLOR,
        0);

      ll = jj + 1;

      // split gemm into tiles
      for (l = j + (2*nb); l < (*m); l += nb) {

        ll++;

        MMM = min(nb,(*m)-l);

        fakedep = (void *)(intptr_t)(j+1);

        sprintf(label, "GEMM %d %d %d", ii, jj, ll);
  
        QUARK_Insert_Task(quark, SCHED_sgemm, 0,
          sizeof(int),             &MMM,     VALUE,
          sizeof(int),             &NN,      VALUE,
          sizeof(int),             &nb,      VALUE,
          sizeof(cuDoubleComplex)*(*m)*(*n), A(l,j),   INPUT,
          sizeof(int),             m,        VALUE,
          sizeof(cuDoubleComplex)*(*m)*(*n), A(j,i),   INPUT,
          sizeof(cuDoubleComplex)*(*m)*(*n), A(l,i),   INOUT,
          sizeof(int),             &priority,VALUE | TASK_PRIORITY,
          sizeof(cuDoubleComplex)*(*m)*(*n), A(i,i),   OUTPUT | GATHERV,
          sizeof(void*),           fakedep,  OUTPUT | GATHERV,
          strlen(label)+1,         label,    VALUE | TASKLABEL,
          5,                       "blue",   VALUE | TASKCOLOR,
          0);

      }

    }

  }

  ii = -1;

  // swap behinds
  for (i = 0; i < k; i += nb) {

    ii++;

    jj = -1;

    MM = min(nb,(*m)-i);
    MM = min(MM,(*n)-i);

    for (j = 0; j < i; j += nb) {

      jj++;

      fakedep = (void *)(intptr_t)(j+1);

      sprintf(label, "LASWPF %d %d", ii, jj);
  
      QUARK_Insert_Task(quark, SCHED_slaswp, 0,
        sizeof(int),             &nb,       VALUE,
        sizeof(cuDoubleComplex)*(*m)*(*n), A(i,j),    INOUT,
        sizeof(int),             m,         VALUE,
        sizeof(int),             &MM,       VALUE,
        sizeof(cuDoubleComplex)*nb,        &ipiv[i],  INPUT,
        sizeof(int),             &priority, VALUE | TASK_PRIORITY,
        sizeof(void*),           fakedep,   INPUT,
        sizeof(cuDoubleComplex)*(*m)*(*n), A(i+nb,j), OUTPUT,
        strlen(label)+1,         label,     VALUE | TASKLABEL,
        7,                       "purple",  VALUE | TASKCOLOR,
        0);

    }

  }

  // synchronization point
  QUARK_Barrier(quark);

  // fix pivot
  ii = -1;

  for (i = 0; i < k; i +=nb) {
    ii++;
    for (j = 0; j < min(nb,(k-i)); j++) {
      ipiv[ii*nb+j] += ii*nb;
    } 
  } 
        
  QUARK_Barrier(quark);

  QUARK_Delete(quark);

  if (TRACE == 1) {
    //dump_trace(4);
  }
}

#undef A


