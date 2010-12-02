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

#include <event_trace.h>

/*

// used by trace 
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

extern "C" Quark *quark;

void SCHED_sgemm(Quark* quark)
{
int UPPER;
int M;
int N;
int K;
cuDoubleComplex *A1;
int LDA;
cuDoubleComplex *A2;
cuDoubleComplex *A3;

cuDoubleComplex mone = MAGMA_Z_NEG_ONE;
cuDoubleComplex one = MAGMA_Z_ONE;

quark_unpack_args_8(quark, UPPER, M, N, K, A1, LDA, A2, A3);

//if (TRACE == 1)
  //core_event_start(QUARK_Thread_Rank(quark));

if (UPPER) {

blasf77_zgemm(MagmaConjTransStr, MagmaNoTransStr, 
  &M, &N, &K, &mone, A1, &LDA, A2, &LDA, &one, A3, &LDA);

} else {

blasf77_zgemm(MagmaNoTransStr, MagmaConjTransStr, 
  &M, &N, &K, &mone, A1, &LDA, A2, &LDA, &one, A3, &LDA);

}


//if (TRACE == 1) {
  //core_event_end(QUARK_Thread_Rank(quark));
  //core_log_event_label(0x0000ff,QUARK_Thread_Rank(quark),0,"");
//}

}

void SCHED_ssyrk(Quark* quark)
{
int UPPER;
int N;
cuDoubleComplex *A1;
int LDA;
int K;
cuDoubleComplex *A2;

//cuDoubleComplex mone = MAGMA_Z_NEG_ONE;
//cuDoubleComplex one = MAGMA_Z_ONE;

double mone = -1.0;
double one = 1.0;

quark_unpack_args_6(quark, UPPER, N, K, A1, LDA, A2);

//if (TRACE == 1)
  //core_event_start(QUARK_Thread_Rank(quark));

if (UPPER) {

//blasf77_zsyrk(MagmaUpperStr, MagmaTransStr, &N, &K, &mone, A1, &LDA, &one, 
blasf77_zherk(MagmaUpperStr, MagmaConjTransStr, &N, &K, &mone, A1, &LDA, &one, 
  A2, &LDA);

} else {

blasf77_zherk(MagmaLowerStr, MagmaNoTransStr, &N, &K, &mone, A1, &LDA, &one, 
  A2, &LDA);

}

//if (TRACE == 1) {
  //core_event_end(QUARK_Thread_Rank(quark));
  //core_log_event_label(0x00ff00,QUARK_Thread_Rank(quark),0,"");
//}

}

void SCHED_spotrf(Quark* quark)
{
int UPPER;
int N;
cuDoubleComplex *A;
int LDA;

int *iinfo;

int info;

quark_unpack_args_5(quark, UPPER, N, A, LDA, iinfo);

//if (TRACE == 1)
  //core_event_start(QUARK_Thread_Rank(quark));

if (UPPER) {

lapackf77_zpotrf(MagmaUpperStr, &N, A, &LDA, &info);

} else {

lapackf77_zpotrf(MagmaLowerStr, &N, A, &LDA, &info);

}

if (info > 0) {
  iinfo[1] = iinfo[0] + info;
}


//if (TRACE == 1) {
  //core_event_end(QUARK_Thread_Rank(quark));
  //core_log_event_label(0x00ffff,QUARK_Thread_Rank(quark),0,"");
//}

}

void SCHED_strsm(Quark* quark)
{
int UPPER;
int M;
int N;
cuDoubleComplex *A1;
int LDA;
cuDoubleComplex *A2;

cuDoubleComplex one = MAGMA_Z_ONE;

quark_unpack_args_6(quark, UPPER, M, N, A1, LDA, A2);

//if (TRACE == 1)
  //core_event_start(QUARK_Thread_Rank(quark));

if (UPPER) {

blasf77_ztrsm(MagmaLeftStr, MagmaUpperStr, MagmaConjTransStr, MagmaNonUnitStr,
  &M, &N, &one, A1, &LDA, A2, &LDA);

} else {

blasf77_ztrsm(MagmaRightStr, MagmaLowerStr, MagmaConjTransStr, MagmaNonUnitStr,
  &M, &N, &one, A1, &LDA, A2, &LDA);

}

//if (TRACE == 1) {
  //core_event_end(QUARK_Thread_Rank(quark));
  ////core_log_event_label(0xff0000,QUARK_Thread_Rank(quark),0,QUARK_Get_Task_Label(quark));
  //core_log_event_label(0xff0000,QUARK_Thread_Rank(quark),0,"");
//}

}

extern "C" int 
magma_zpotrf_mc(
char *uplo,
int *n,
cuDoubleComplex *a,
int *lda,
int *info)
{
/*  -- MAGMA (version 0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2009

    Purpose   
    =======   

    SPOTRF computes the Cholesky factorization of a real symmetric   
    positive definite matrix A.   

    The factorization has the form   
       A = U**T * U,  if UPLO = 'U', or   
       A = L  * L**T,  if UPLO = 'L',   
    where U is an upper triangular matrix and L is lower triangular.   

    This is the block version of the algorithm, calling Level 3 BLAS.   

    Arguments   
    =========   

    UPLO    (input) CHARACTER*1   
            = 'U':  Upper triangle of A is stored;   
            = 'L':  Lower triangle of A is stored.   

    N       (input) INTEGER   
            The order of the matrix A.  N >= 0.   

    A       (input/output) REAL array, dimension (LDA,N)   
            On entry, the symmetric matrix A.  If UPLO = 'U', the leading   
            N-by-N upper triangular part of A contains the upper   
            triangular part of the matrix A, and the strictly lower   
            triangular part of A is not referenced.  If UPLO = 'L', the   
            leading N-by-N lower triangular part of A contains the lower   
            triangular part of the matrix A, and the strictly upper   
            triangular part of A is not referenced.   

            On exit, if INFO = 0, the factor U or L from the Cholesky   
            factorization A = U**T*U or A = L*L**T.   

    LDA     (input) INTEGER   
            The leading dimension of the array A.  LDA >= max(1,N).   

    INFO    (output) INTEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument had an illegal value   
            > 0:  if INFO = i, the leading minor of order i is not   
                  positive definite, and the factorization could not be   
                  completed.   

    =====================================================================   */

  // check arguments
  int upper = (int) lsame_(uplo, "U");                                          
  *info = 0;
  if (! upper && ! lsame_(uplo, "L")) {
    *info = -1;
  } else if (*n < 0) {
    *info = -2;
  } else if (*lda < max(1,*n)) {
    *info = -4;
  }
  if (*info != 0)
    return 0;

  // get block size
  int nb = EN_BEE;

  int i,j,k;
  int ii,jj,kk;
  int temp,temp2,temp3;

  char label[10000];

  int iinfo[2];
  iinfo[1] = 0;

  // start quark with 4 threads
  //Quark *quark = QUARK_New(0);

  ii = -1;

  // traverse diagonal blocks
  for (i = 0; i < (*n); i += nb) {

    ii++;

    temp2 = min(nb,(*n)-i);
 
    // if not first block
    if (i > 0) {

      // first do large syrk, then split
      if (i < (*n)/2) {

        sprintf(label, "SYRK %d", ii);

        if (upper) {

          QUARK_Insert_Task(quark, SCHED_ssyrk, 0,
            sizeof(int),             &upper,    VALUE,
            sizeof(int),             &temp2,    VALUE,
            sizeof(int),             &i,        VALUE,
            sizeof(cuDoubleComplex)*(*n)*(*n), A(0,i),    INPUT,
            sizeof(int),             lda,       VALUE,
            sizeof(cuDoubleComplex)*(*n)*(*n), A(i,i),    INOUT,
            sizeof(cuDoubleComplex)*(*n)*(*n), A(i-nb,i), INPUT,
            strlen(label)+1,         label,     VALUE | TASKLABEL,
            6,                       "green",   VALUE | TASKCOLOR,
            0);

        } else {

          QUARK_Insert_Task(quark, SCHED_ssyrk, 0,
            sizeof(int),             &upper,    VALUE,
            sizeof(int),             &temp2,    VALUE,
            sizeof(int),             &i,        VALUE,
            sizeof(cuDoubleComplex)*(*n)*(*n), A(i,0),    INPUT,
            sizeof(int),             lda,       VALUE,
            sizeof(cuDoubleComplex)*(*n)*(*n), A(i,i),    INOUT,
            sizeof(cuDoubleComplex)*(*n)*(*n), A(i,i-nb), INPUT,
            strlen(label)+1,         label,     VALUE | TASKLABEL,
            6,                       "green",   VALUE | TASKCOLOR,
            0);

        }   

      } else {

        jj = -1;

        // split syrk into tiles
        for (j = 0; j < i; j += nb) {

          jj++;

          sprintf(label, "SYRK %d %d", ii, jj);

          if (upper) {

            QUARK_Insert_Task(quark, SCHED_ssyrk, 0,
              sizeof(int),             &upper,    VALUE,
              sizeof(int),             &temp2,    VALUE,
              sizeof(int),             &nb,       VALUE,
              sizeof(cuDoubleComplex)*(*n)*(*n), A(j,i),    INPUT,
              sizeof(int),             lda,       VALUE,
              sizeof(cuDoubleComplex)*(*n)*(*n), A(i,i),    INOUT,
              strlen(label)+1,         label,     VALUE | TASKLABEL,
              6,                       "green",   VALUE | TASKCOLOR,
              0);

          } else {

            QUARK_Insert_Task(quark, SCHED_ssyrk, 0,
              sizeof(int),             &upper,    VALUE,
              sizeof(int),             &temp2,    VALUE,
              sizeof(int),             &nb,       VALUE,
              sizeof(cuDoubleComplex)*(*n)*(*n), A(i,j),    INPUT,
              sizeof(int),             lda,       VALUE,
              sizeof(cuDoubleComplex)*(*n)*(*n), A(i,i),    INOUT,
              strlen(label)+1,         label,     VALUE | TASKLABEL,
              6,                       "green",   VALUE | TASKCOLOR,
              0);

          }

        }

      }

      // if not last block
      if (i < ((*n)-nb)) {

        jj = -1;

        // split gemm into tiles
        for (j = i+nb; j < (*n); j += nb){

          jj++;

          kk = -1;

          for (k = 0; k < i; k += nb) {

            kk++;

            temp = min(nb,(*n)-j);

            sprintf(label, "GEMM %d %d %d", ii, jj, kk);

            if (upper) {

              QUARK_Insert_Task(quark, SCHED_sgemm, 0,
                sizeof(int),             &upper,    VALUE,
                sizeof(int),             &nb,       VALUE,
                sizeof(int),             &temp,     VALUE,
                sizeof(int),             &nb,       VALUE,
                sizeof(cuDoubleComplex)*(*n)*(*n), A(k,i), INPUT,
                sizeof(int),             lda,       VALUE,
                sizeof(cuDoubleComplex)*(*n)*(*n), A(k,j),    INPUT,
                sizeof(cuDoubleComplex)*(*n)*(*n), A(i,j), INOUT,
                strlen(label)+1,         label,     VALUE | TASKLABEL,
                5,                       "blue",    VALUE | TASKCOLOR,
                0);

            } else {

              QUARK_Insert_Task(quark, SCHED_sgemm, 0,
                sizeof(int),             &upper,    VALUE,
                sizeof(int),             &temp,     VALUE,
                sizeof(int),             &nb,       VALUE,
                sizeof(int),             &nb,       VALUE,
                sizeof(cuDoubleComplex)*(*n)*(*n), A(j,k), INPUT,
                sizeof(int),             lda,       VALUE,
                sizeof(cuDoubleComplex)*(*n)*(*n), A(i,k),    INPUT,
                sizeof(cuDoubleComplex)*(*n)*(*n), A(j,i), INOUT,
                strlen(label)+1,         label,     VALUE | TASKLABEL,
                5,                       "blue",    VALUE | TASKCOLOR,
                0);

            }

          }

        }

	  }

	}

    iinfo[0] = i;

    sprintf(label, "POTRF %d", ii);

    QUARK_Insert_Task(quark, SCHED_spotrf, 0,
      sizeof(int),             &upper,    VALUE,
      sizeof(int),             &temp2,    VALUE,
      sizeof(cuDoubleComplex)*(*n)*(*n), A(i,i),    INOUT,
      sizeof(int),             lda,       VALUE,
      sizeof(int),             iinfo,     OUTPUT,
      strlen(label)+1,         label,     VALUE | TASKLABEL,
      5,                       "cyan",    VALUE | TASKCOLOR,
      0);

    // if not last block
    if (i < ((*n)-nb)) {

      // split trsm into tiles
      for (j = i + nb; j < (*n); j += nb) {

        temp = min(nb,(*n)-j);

        sprintf(label, "TRSM %d", ii);

        if (upper) {

          QUARK_Insert_Task(quark, SCHED_strsm, 0,
            sizeof(int),             &upper,    VALUE,
            sizeof(int),             &nb,       VALUE,
            sizeof(int),             &temp,     VALUE,
            sizeof(cuDoubleComplex)*(*n)*(*n), A(i,i),    INPUT,
            sizeof(int),             lda,       VALUE,
            sizeof(cuDoubleComplex)*(*n)*(*n), A(i,j),    INOUT,
            strlen(label)+1,         label,     VALUE | TASKLABEL,
            4,                       "red",     VALUE | TASKCOLOR,
            0);

        } else {

          QUARK_Insert_Task(quark, SCHED_strsm, 0,
            sizeof(int),             &upper,    VALUE,
            sizeof(int),             &temp,     VALUE,
            sizeof(int),             &nb,       VALUE,
            sizeof(cuDoubleComplex)*(*n)*(*n), A(i,i),    INPUT,
            sizeof(int),             lda,       VALUE,
            sizeof(cuDoubleComplex)*(*n)*(*n), A(j,i),    INOUT,
            strlen(label)+1,         label,     VALUE | TASKLABEL,
            4,                       "red",     VALUE | TASKCOLOR,
            0);

        }

	  }

    }

  }

  QUARK_Barrier(quark);

  //QUARK_Delete(quark);

  if (TRACE == 1) {
    //dump_trace(4);
  }

}

#undef A

