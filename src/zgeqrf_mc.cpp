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

*/

/*
if (tag != 0){
  char dkdk[10];
  sprintf(dkdk,"%d",tag);

  fprintf(trace_file,
  "    "
  "<text x=\"%.2lf\" y=\"%.0lf\" font-size=\"20\" fill=\"black\">"
  "%s"
  "</text>\n",
  start * scale + 10, // x
  core * 100.0 + 20, // y
  dkdk);
}
*/


/*


}
	
fprintf(trace_file,
"  </g>\n"
"</svg>\n");
	
fclose(trace_file);
}

*/


#define  A(m,n) (a+(n)*(*lda)+(m))
#define  T(m) (work+(m)*(nb))
#define  W(k,n) &(local_work[(mt)*(n-1)+(k)])

extern "C" int EN_BEE;

extern "C" int TRACE;

void sgetro (char *trans, const int m, const int n, const cuDoubleComplex *A, const int LDA, cuDoubleComplex *B, const int LDB) 
{
  const cuDoubleComplex *Atmp;

  int i, j;

  for (i=0; i<m; i++) {
    Atmp = A + i;
    for (j=0; j<n; j++) {

      if (trans[0] == 'C') {

        double *ap,*tp;
        cuDoubleComplex tmp;
        ap = (double *)&(*Atmp);
        tp = (double *)&tmp;
        tp[0] = ap[0];
        tp[1] = -ap[1];

        *B = tmp;

      } else {

        *B = *Atmp;

      }

      B += 1;
      Atmp += LDA;
    }
    B += (LDB - n);
  }
}

void SCHED_slarfb(Quark* quark)
{

int M;
int N;
int MM;
int NN;
int IB;
int LDV;
int LDC;
int LDT;
int LDW;

cuDoubleComplex *V;
cuDoubleComplex *C;
cuDoubleComplex *T;
cuDoubleComplex **W;

quark_unpack_args_13(quark, M, N, MM, NN, IB, V, LDV, C, LDC, T, LDT, W, LDW);

//fprintf(stderr,"slarfb\n");

if (M < 0) {
  printf("SCHED_slarfb:  illegal value of M\n");
}
if (N < 0) {
  printf("SCHED_slarfb:  illegal value of N\n");
}
if (IB < 0) {
  printf("SCHED_slarfb:  illegal value of IB\n");
}

*W = (cuDoubleComplex*) malloc(LDW*MM*sizeof(cuDoubleComplex));

  //if (TRACE == 1)
    //core_event_start(QUARK_Thread_Rank(quark));

sgetro(MagmaConjTransStr, MM, NN, C, LDC, *W, LDW);

cuDoubleComplex c_one = MAGMA_Z_ONE;

blasf77_ztrmm("r","l","n","u",
  &NN, &MM, &c_one, V, &LDV, *W, &LDW);

int K=M-MM;

blasf77_zgemm(MagmaConjTransStr, "n", &NN, &MM, &K,
  &c_one, &C[MM], &LDC, &V[MM], &LDV, &c_one, *W, &LDW);

blasf77_ztrmm("r", "u", "n", "n", 
  &NN, &MM, &c_one, T, &LDT, *W, &LDW);

/*

if (TRACE == 1) {
  core_event_end(QUARK_Thread_Rank(quark));
  core_log_event(0x00FFFF, QUARK_Thread_Rank(quark), 0);
}

*/

}

void SCHED_sgeqrt(Quark* quark)
{
int M;
int N;
int IB;
cuDoubleComplex *A;
int LDA;
cuDoubleComplex *T;
int LDT;
cuDoubleComplex *TAU;
cuDoubleComplex *WORK;

int iinfo;
int lwork=-1;

quark_unpack_args_9(quark, M, N, IB, A, LDA, T, LDT, TAU, WORK);

//fprintf(stderr,"sgeqrt\n");

if (M < 0) { 
  printf("SCHED_sgeqrt: illegal value of M\n");
}

if (N < 0) { 
  printf("SCHED_sgeqrt: illegal value of N\n");
}

if ((IB < 0) || ( (IB == 0) && ((M > 0) && (N > 0)) )) {
  printf("SCHED_sgeqrt: illegal value of IB\n");
}

if ((LDA < max(1,M)) && (M > 0)) {
  printf("SCHED_sgeqrt: illegal value of LDA\n");
}

if ((LDT < max(1,IB)) && (IB > 0)) {
  printf("SCHED_sgeqrt: illegal value of LDT\n");
}

  //if (TRACE == 1)
    //core_event_start(QUARK_Thread_Rank(quark));

//lapackf77_zgeqrf(&M, &N, A, &LDA, TAU, WORK, &lwork, &iinfo);
//lwork=(int)WORK[0];
lwork=N;
lapackf77_zgeqrf(&M, &N, A, &LDA, TAU, WORK, &lwork, &iinfo);

lapackf77_zlarft("F", "C", &M, &N, A, &LDA, TAU, T, &LDT);

//if (TRACE == 1) {
  //core_event_end(QUARK_Thread_Rank(quark));
  //core_log_event(0x00FF00, QUARK_Thread_Rank(quark), 0);
//}

}

void SCHED_strmm(Quark *quark)
{
  int m;
  int n;
  cuDoubleComplex alpha;
  cuDoubleComplex *a;
  int lda;
  cuDoubleComplex **b;
  int ldb;
  cuDoubleComplex beta;
  cuDoubleComplex *c;
  int ldc;
  cuDoubleComplex *work;

  int j;

  int one = 1;

  quark_unpack_args_11(quark, m, n, alpha, a, lda, b, ldb, beta, c, ldc, work);

//fprintf(stderr,"strmm\n");

  if (m < 0) {
    printf("SCHED_strmm:  illegal value of m\n");
  }

  if (n < 0) {
    printf("SCHED_strmm:  illegal value of n\n");
  }

  //if (TRACE == 1)
    //core_event_start(QUARK_Thread_Rank(quark));

  sgetro(MagmaConjTransStr, n, m, *b, ldb, work, m);

  blasf77_ztrmm("l", "l", "n", "u", 
    &m, &n, &alpha, a, &lda, work, &m);

  for (j = 0; j < n; j++)
  {
    blasf77_zaxpy(&m, &beta, &(work[j*m]), &one, &(c[j*ldc]), &one);
  }

  //if (TRACE == 1) {
    //core_event_end(QUARK_Thread_Rank(quark));
    //core_log_event(0xFFA000, QUARK_Thread_Rank(quark), 0);
  //}

}

void SCHED_sgemm(Quark *quark)
{
  int m;
  int n;
  int k;
  cuDoubleComplex alpha;
  cuDoubleComplex *a;
  int lda;
  cuDoubleComplex **b;
  int ldb;
  cuDoubleComplex beta;
  cuDoubleComplex *c;
  int ldc;
  
  cuDoubleComplex *fake;

  int dkdk;

  quark_unpack_args_13(quark, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, fake, dkdk);
      
//fprintf(stderr,"gemm\n");

  //if (TRACE == 1)
    //core_event_start(QUARK_Thread_Rank(quark));

  blasf77_zgemm("n", MagmaConjTransStr, 
    &m, &n, &k, &alpha, a, &lda, *b, &ldb, &beta, c, &ldc);

  //if (TRACE == 1) {
    //core_event_end(QUARK_Thread_Rank(quark));
    //core_log_event(0x9932CC, QUARK_Thread_Rank(quark), 0);
  //}

}

void QUARK_Insert_Task_sgemm(Quark *quark, Quark_Task_Flags *task_flags, 
  int m, 
  int n, 
  int k,
  cuDoubleComplex alpha,
  cuDoubleComplex *a,
  int lda,
  cuDoubleComplex **b,
  int ldb,
  cuDoubleComplex beta,
  cuDoubleComplex *c,
  int ldc,
  cuDoubleComplex *fake,
  char *dag_label,
  int priority, 
  int dkdk)
{

  QUARK_Insert_Task(quark, SCHED_sgemm, task_flags,
    sizeof(int),           &m,     VALUE,
    sizeof(int),           &n,     VALUE,
    sizeof(int),           &k,     VALUE,
    sizeof(cuDoubleComplex),         &alpha, VALUE,
    sizeof(cuDoubleComplex)*ldb*ldb, a,      INPUT,
    sizeof(int),           &lda,   VALUE,
    sizeof(cuDoubleComplex*),        b,      INPUT,
    sizeof(int),           &ldb,   VALUE,
    sizeof(cuDoubleComplex),         &beta,  VALUE,
    sizeof(cuDoubleComplex)*ldb*ldb, c,      INOUT | LOCALITY,
    sizeof(int),           &ldc,   VALUE,
    sizeof(cuDoubleComplex)*ldb*ldb, fake,   OUTPUT | GATHERV,
    sizeof(int),           &priority, VALUE | TASK_PRIORITY,
    sizeof(int),&dkdk,VALUE,
    strlen(dag_label)+1,   dag_label, VALUE | TASKLABEL,
6,                     "purple",   VALUE | TASKCOLOR,
    0);
}

void QUARK_Insert_Task_strmm(Quark *quark, Quark_Task_Flags *task_flags,
  int m,
  int n,
  cuDoubleComplex alpha, 
  cuDoubleComplex *a,
  int lda,
  cuDoubleComplex **b,
  int ldb,
  cuDoubleComplex beta,
  cuDoubleComplex *c,
  int ldc,
  char *dag_label,
  int priority)
{

  QUARK_Insert_Task(quark, SCHED_strmm, task_flags,
    sizeof(int),           &m,     VALUE,
    sizeof(int),           &n,     VALUE,
    sizeof(cuDoubleComplex),         &alpha, VALUE,
    sizeof(cuDoubleComplex)*ldb*ldb, a,      INPUT,
    sizeof(int),           &lda,   VALUE,
    sizeof(cuDoubleComplex*),        b,      INPUT,
    sizeof(int),           &ldb,   VALUE,
    sizeof(cuDoubleComplex),         &beta,  VALUE,
    sizeof(cuDoubleComplex)*ldb*ldb, c,      INOUT | LOCALITY,
    sizeof(int),           &ldc,   VALUE,
    sizeof(cuDoubleComplex)*ldb*ldb, NULL,   SCRATCH,
    sizeof(int),           &priority, VALUE | TASK_PRIORITY,
    strlen(dag_label)+1,   dag_label, VALUE | TASKLABEL,
    6,                     "orange",   VALUE | TASKCOLOR,
    0);
}

void QUARK_Insert_Task_sgeqrt(Quark *quark, Quark_Task_Flags *task_flags, 
  int m,
  int n,
  cuDoubleComplex *a,
  int lda,
  cuDoubleComplex *t,
  int ldt,
  cuDoubleComplex *tau,
  char *dag_label)
{

  int priority = 1000;

  QUARK_Insert_Task(quark, SCHED_sgeqrt, task_flags,
    sizeof(int),           &m,        VALUE,
    sizeof(int),           &n,        VALUE,
    sizeof(int),           &ldt,      VALUE,
    sizeof(cuDoubleComplex)*m*n,     a,         INOUT | LOCALITY,
    sizeof(int),           &lda,      VALUE,
    sizeof(cuDoubleComplex)*ldt*ldt, t,         OUTPUT,
    sizeof(int),           &ldt,      VALUE,
    sizeof(cuDoubleComplex)*ldt,     tau,       OUTPUT,
    sizeof(cuDoubleComplex)*ldt*ldt, NULL,      SCRATCH,
    sizeof(int),           &priority, VALUE | TASK_PRIORITY,
    strlen(dag_label)+1,   dag_label, VALUE | TASKLABEL,
    6,                     "green",   VALUE | TASKCOLOR,
    0);

}

void QUARK_Insert_Task_slarfb(Quark *quark, Quark_Task_Flags *task_flags,
  int m,
  int n,
  int mm,
  int nn,
  int ib,
  cuDoubleComplex *v,
  int ldv,
  cuDoubleComplex *c,
  int ldc,
  cuDoubleComplex *t,
  int ldt,
  cuDoubleComplex **w,
  int ldw,
  char *dag_label,
  int priority)

{

  QUARK_Insert_Task(quark, SCHED_slarfb, task_flags,
    sizeof(int),         &m,        VALUE,
    sizeof(int),         &n,        VALUE,
    sizeof(int),         &mm,       VALUE,
    sizeof(int),         &nn,       VALUE,
    sizeof(int),         &ib,       VALUE,
    sizeof(cuDoubleComplex)*m*n,   v,         INPUT,
    sizeof(int),         &ldv,      VALUE,
    sizeof(cuDoubleComplex)*m*n,   c,         INPUT,
    sizeof(int),         &ldc,      VALUE,
    sizeof(cuDoubleComplex)*ib*ib, t,         INPUT,
    sizeof(int),         &ldt,      VALUE,
    sizeof(cuDoubleComplex*),      w,         OUTPUT | LOCALITY,
    sizeof(int),         &ldw,      VALUE,
    sizeof(int),         &priority, VALUE | TASK_PRIORITY,
    strlen(dag_label)+1, dag_label, VALUE | TASKLABEL,
    6,                   "cyan",    VALUE | TASKCOLOR,
    0);

}

extern "C" int 
magma_zgeqrf_mc( magma_int_t *m, magma_int_t *n,
                 cuDoubleComplex *a,    magma_int_t *lda, cuDoubleComplex *tau,
                 cuDoubleComplex *work, magma_int_t *lwork,
                 magma_int_t *info)
{
/*  -- MAGMA (version 0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2009

    Purpose   
    =======   

    SGEQRF computes a QR factorization of a real M-by-N matrix A:   
    A = Q * R.   

    Arguments   
    =========   

    M       (input) INTEGER   
            The number of rows of the matrix A.  M >= 0.   

    N       (input) INTEGER   
            The number of columns of the matrix A.  N >= 0.   

    A       (input/output) REAL array, dimension (LDA,N)   
            On entry, the M-by-N matrix A.   
            On exit, the elements on and above the diagonal of the array   
            contain the min(M,N)-by-N upper trapezoidal matrix R (R is   
            upper triangular if m >= n); the elements below the diagonal,   
            with the array TAU, represent the orthogonal matrix Q as a   
            product of min(m,n) elementary reflectors (see Further   
            Details).   

    LDA     (input) INTEGER   
            The leading dimension of the array A.  LDA >= max(1,M).   

    TAU     (output) REAL array, dimension (min(M,N))   
            The scalar factors of the elementary reflectors (see Further   
            Details).   

    WORK    (workspace/output) REAL array, dimension (MAX(1,LWORK))   
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.   

    LWORK   (input) INTEGER   
            The dimension of the array WORK.  LWORK >= N*NB. 

            If LWORK = -1, then a workspace query is assumed; the routine   
            only calculates the optimal size of the WORK array, returns   
            this value as the first entry of the WORK array, and no error   
            message related to LWORK is issued.

    INFO    (output) INTEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument had an illegal value   

    Further Details   
    ===============   

    The matrix Q is represented as a product of elementary reflectors   

       Q = H(1) H(2) . . . H(k), where k = min(m,n).   

    Each H(i) has the form   

       H(i) = I - tau * v * v'   

    where tau is a real scalar, and v is a real vector with   
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),   
    and tau in TAU(i).   

    ====================================================================    */

  int i,j,l;

  int ii=-1,jj=-1,ll=-1;

  char sgeqrt_dag_label[1000]; 
  char slarfb_dag_label[1000];
  char strmm_dag_label[1000];
  char sgemm_dag_label[1000];

  *info = 0;

  cuDoubleComplex c_one = MAGMA_Z_ONE;
  cuDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;

  int nb = EN_BEE;

  int lwkopt = *n * nb;
  work[0] = MAGMA_Z_MAKE( (double)lwkopt, 0 );

  long int lquery = *lwork == -1;

  if (*m < 0) {
    *info = -1;
  } else if (*n < 0) {
    *info = -2;
  } else if (*lda < max(1,*m)) {
    *info = -4;
  } else if (*lwork < max(1,*n) && ! lquery) {
    *info = -7;
  }
  if (*info != 0)
    return 0;
  else if (lquery)
    return 0;

  int k = min(*m,*n);
  if (k == 0) {
    work[0] = c_one;
    return 0;
  }

  int nt = (((*n)%nb) == 0) ? (*n)/nb : (*n)/nb + 1;
  int mt = (((*m)%nb) == 0) ? (*m)/nb : (*m)/nb + 1;

  cuDoubleComplex **local_work = (cuDoubleComplex**) malloc(sizeof(cuDoubleComplex*)*(nt-1)*mt);
  memset(local_work, 0, sizeof(cuDoubleComplex*)*(nt-1)*mt);

  Quark *quark;
  quark = QUARK_New(4);

  int priority;

  for (i = 0; i < k; i += nb) {

    ii++;

    jj = ii;

    sprintf(sgeqrt_dag_label, "GEQRT %d",ii);

    QUARK_Insert_Task_sgeqrt(quark, 
      0, (*m)-i, min(nb,(*n)-i), A(i,i), *m, T(i), nb, &tau[i], sgeqrt_dag_label);

    if (i > 0) {

      priority = 100;

      for (j = (i-nb) + (2*nb); j < *n; j += nb) { 

        jj++;

        ll = ii-1;

        sprintf(slarfb_dag_label, "LARFB %d %d",ii-1, jj);

        QUARK_Insert_Task_slarfb(quark, 0, 
          (*m)-(i-nb), min(nb,(*n)-(i-nb)), min(nb,(*m)-(i-nb)), min(nb,(*n)-j), nb, 
          A(i-nb,i-nb), *m, A(i-nb,j), *m, T(i-nb), nb, W(ii-1,jj), nb, slarfb_dag_label, priority);

        sprintf(strmm_dag_label, "TRMM %d %d",ii-1, jj);

        QUARK_Insert_Task_strmm(quark, 0, min(nb,(*m)-(i-nb)), min(nb,(*n)-j), c_neg_one, 
          A(i-nb,i-nb), *m, W(ii-1,jj), nb, c_one, A(i-nb,j), *m, strmm_dag_label, priority);

          sprintf(sgemm_dag_label, "GEMM %d %d %d",ii-1, jj, ll);

          QUARK_Insert_Task_sgemm(quark, 0, (*m)-i, min(nb,(*n)-j), min(nb,(*n)-(i-nb)), c_neg_one,
            A(i,i-nb), *m, W(ii-1,jj), nb, c_one, A(i,j), *m, A(i,j), sgemm_dag_label, priority, jj);

      }

    }

    j = i + nb;

    jj = ii;

    if (j < (*n)) {

      priority = 0;

      jj++;

      ll = ii;

      sprintf(slarfb_dag_label, "LARFB %d %d",ii, jj);

      QUARK_Insert_Task_slarfb(quark, 0, 
        (*m)-i, min(nb,(*n)-i), min(nb,(*m)-i), min(nb,(*n)-j), nb, 
        A(i,i), *m, A(i,j), *m, T(i), nb, W(ii,jj), nb, slarfb_dag_label, priority);

      sprintf(strmm_dag_label, "TRMM %d %d",ii, jj);

      QUARK_Insert_Task_strmm(quark, 0, min(nb,(*m)-i), min(nb,(*n)-j), c_neg_one, 
        A(i,i), *m, W(ii,jj), nb, c_one, A(i,j), *m, strmm_dag_label, priority);

        sprintf(sgemm_dag_label, "GEMM %d %d %d",ii, jj, ll);

        QUARK_Insert_Task_sgemm(quark, 0, (*m)-i-nb, min(nb,(*n)-j), min(nb,(*n)-i), c_neg_one,
          A(i+nb,i), *m, W(ii,jj), nb, c_one, A(i+nb,j), *m, A(i+nb,j), sgemm_dag_label, priority, jj);

    }

  }

  QUARK_Barrier(quark);

  QUARK_Delete(quark);

  for(k = 0 ; k < (nt-1)*mt; k++) {
    if (local_work[k] != NULL) {
      free(local_work[k]);
}
  }
  free(local_work);

  if (TRACE == 1) {
    //dump_trace(4);
  }

}

#undef A
#undef T
#undef W


