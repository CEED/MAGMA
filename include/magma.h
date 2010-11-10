/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010
*/

#ifndef _MAGMA_
#define _MAGMA_

/* 
 * MAGMA Blas Functions 
 */ 
#include "magmablas.h"

#include "auxiliary.h"
#include "magma_lapack.h"

/*
 * MAGMA functions
 */
#include "magma_z.h"
#include "magma_c.h"
#include "magma_d.h"
#include "magma_s.h"
#include "magma_zc.h"
#include "magma_ds.h"

#define MAGMA_S_ZERO 0.0f
#define MAGMA_S_ONE 1.0f
#define MAGMA_S_NEG_ONE -1.0f
#define MAGMA_D_ZERO 0.0
#define MAGMA_D_ONE 1.0
#define MAGMA_D_NEG_ONE -1.0f
#define MAGMA_C_ZERO {0.0f, 0.0f}
#define MAGMA_C_ONE {1.0f, 0.0f}
#define MAGMA_C_NEG_ONE {-1.0f, 0.0f}
#define MAGMA_Z_ZERO {0.0, 0.0}
#define MAGMA_Z_ONE {1.0, 0.0}
#define MAGMA_Z_NEG_ONE {-1.0, 0.0}

#define MAGMA_Z_SET2REAL(v, t) v.x = (t); v.y = 0.0
#define MAGMA_Z_OP_NEG_ASGN(t, z) (t).x = -(z).x; (t).y = -(z).y
#define MAGMA_Z_EQUAL(u,v) (((u).x == (v).x) && ((u).y == (v).y))
#define MAGMA_Z_GET_X(u) ((u).x)

#define MAGMA_C_SET2REAL(v, t) v.x = (t); v.y = 0.0
#define MAGMA_C_OP_NEG_ASGN(t, z) (t).x = -(z).x; (t).y = -(z).y
#define MAGMA_C_EQUAL(u,v) (((u).x == (v).x) && ((u).y == (v).y))
#define MAGMA_C_GET_X(u) ((u).x)

#define MAGMA_D_SET2REAL(v, t) v = (t);
#define MAGMA_D_OP_NEG_ASGN(t, z) (t) = -(z)
#define MAGMA_D_EQUAL(u,v) ((u) == (v))
#define MAGMA_D_GET_X(u) (u)

#define MAGMA_S_SET2REAL(v, t) v = (t);
#define MAGMA_S_OP_NEG_ASGN(t, z) (t) = -(z)
#define MAGMA_S_EQUAL(u,v) ((u) == (v))
#define MAGMA_S_GET_X(u) (u)

/* Missing CUDA BLAS */
extern void cublasZsymv(char uplo, int n, double2 alpha, double2 *a, int lda,
                      double2 *x, int incx, double2 beta, double2 *y, int incy);

extern void cublasCsymv(char uplo, int n, float2 alpha, float2 *a, int lda,
                        float2 *x, int incx, float2 beta, float2 *y, int incy);


#ifdef __cplusplus
extern "C" {
#endif

void magmablas_sdlaswp(int, double *, int, float *, int, int *);


/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA function definitions
*/
void magma_xerbla(char *name , magma_int_t *info);

#ifdef __cplusplus
}
#endif

#endif

