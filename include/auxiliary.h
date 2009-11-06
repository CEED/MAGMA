/*
    -- MAGMA (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       June 2009
*/

#ifndef _MAGMA_AUXILIARY_
#define _MAGMA_AUXILIARY_

int magma_get_spotrf_nb(int m);
int magma_get_sgeqrf_nb(int m);
int magma_get_sgeqlf_nb(int m);
int magma_get_sgetrf_nb(int m);
int magma_get_sgehrd_nb(int m);
int magma_get_sgelqf_nb(int m);

int magma_get_dpotrf_nb(int m);
int magma_get_dgeqrf_nb(int m);
int magma_get_dgetrf_nb(int m);
int magma_get_dgehrd_nb(int m);

int magma_get_cpotrf_nb(int m);
int magma_get_cgetrf_nb(int m);
int magma_get_zgetrf_nb(int m);

#include <sys/time.h>
typedef struct timestruct
{
  unsigned int sec;
  unsigned int usec;
} TimeStruct;

TimeStruct get_current_time(void);
double GetTimerValue(TimeStruct time_1, TimeStruct time_2);

void printout_devices();

void spanel_to_q(char uplo, int ib, float *a, int lda, float *work);
void sq_to_panel(char uplo, int ib, float *a, int lda, float *work);

void swp2pswp(int n, int *ipiv, int *newipiv);

void cpanel_to_q(int ib, float2 *a, int lda, float2 *work);
void cq_to_panel(int ib, float2 *a, int lda, float2 *work);

void dpanel_to_q(int ib, double *a, int lda, double *work);
void dq_to_panel(int ib, double *a, int lda, double *work);

#endif
