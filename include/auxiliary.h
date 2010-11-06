/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010
*/

#ifndef _MAGMA_AUXILIARY_
#define _MAGMA_AUXILIARY_

#include <sys/time.h>
typedef struct timestruct
{
  unsigned int sec;
  unsigned int usec;
} TimeStruct;

#ifdef __cplusplus
extern "C" {
#endif

int magma_get_spotrf_nb(int m);
int magma_get_sgeqrf_nb(int m);
int magma_get_sgeqlf_nb(int m);
int magma_get_sgetrf_nb(int m);
int magma_get_sgehrd_nb(int m);
int magma_get_ssytrd_nb(int m);
int magma_get_sgelqf_nb(int m);
int magma_get_sgebrd_nb(int m);

int magma_get_dpotrf_nb(int m);
int magma_get_dgeqrf_nb(int m);
int magma_get_dgeqlf_nb(int m);
int magma_get_dgetrf_nb(int m);
int magma_get_dgehrd_nb(int m);
int magma_get_dsytrd_nb(int m);
int magma_get_dgelqf_nb(int m);
int magma_get_dgebrd_nb(int m);

int magma_get_cpotrf_nb(int m);
int magma_get_cgetrf_nb(int m);
int magma_get_cgeqrf_nb(int m);
int magma_get_cgeqlf_nb(int m);
int magma_get_cgehrd_nb(int m);
int magma_get_cgelqf_nb(int m);

int magma_get_zpotrf_nb(int m);
int magma_get_zgetrf_nb(int m);
int magma_get_zgeqrf_nb(int m);
int magma_get_zgeqlf_nb(int m);
int magma_get_zgehrd_nb(int m);
int magma_get_zgelqf_nb(int m);

TimeStruct get_current_time(void);
double GetTimerValue(TimeStruct time_1, TimeStruct time_2);

void printout_devices();

void spanel_to_q(char uplo, int ib, float *a, int lda, float *work);
void sq_to_panel(char uplo, int ib, float *a, int lda, float *work);

void swp2pswp(int n, int *ipiv, int *newipiv);

void cpanel_to_q(char uplo, int ib, float2 *a, int lda, float2 *work);
void cq_to_panel(char uplo, int ib, float2 *a, int lda, float2 *work);

void dpanel_to_q(char uplo, int ib, double *a, int lda, double *work);
void dq_to_panel(char uplo, int ib, double *a, int lda, double *work);

void zpanel_to_q(char uplo, int ib, double2 *a, int lda, double2 *work);
void zq_to_panel(char uplo, int ib, double2 *a, int lda, double2 *work);

float getv(float *da);

#ifdef __cplusplus
}
#endif

#endif
