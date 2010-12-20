/**
 *
 * @file flops.h
 *
 *  File provided by Univ. of Tennessee,
 *
 * @version 1.0.0
 * @author Mathieu Faverge
 * @date 2010-12-20
 *
 **/
/*
 * This file provide the flops formula for all Level 3 BLAS and some
 * Lapack routines.  Each macro uses the same size parameters as the
 * function associated and provide one formula for additions and one
 * for multiplications. Ecample to use these macros:
 *  - In real:
 *    flops = FMULS_GEMM((double)m, (double(n), (double(k)) 
 *          + FADDS_GEMM((double)m, (double(n), (double(k));
 *
 *  - In complex:
 *    flops = 6.0 * FMULS_GEMM((double)m, (double(n), (double(k)) 
 *          + 2.0 * FADDS_GEMM((double)m, (double(n), (double(k));
 *
 * All the formula are reported in the LAPACK Lawn 41:
 *     http://www.netlib.org/lapack/lawns/lawn41.ps
 */
#ifndef _FLOPS_H_
#define _FLOPS_H_

/*
 * Level 3 BLAS 
 */  
#define FADDS_GEMM(m, n, k) ((m) * (n) * (k))
#define FMULS_GEMM(m, n, k) ((m) * (n) * (k))

#define FADDS_SYMM_L(m, n) ((m) * (m) * (n))
#define FMULS_SYMM_L(m, n) ((m) * (m) * (n))

#define FADDS_SYMM_R(m, n) ((m) * (n) * (n))
#define FMULS_SYMM_R(m, n) ((m) * (n) * (n))

#define FADDS_SYRK(k, n) (0.5 * (k) * (n) * ((n)+1))
#define FMULS_SYRK(k, n) (0.5 * (k) * (n) * ((n)+1))

#define FADDS_SYR2K(k, n) ((k) * (n) * (n) + (n))
#define FMULS_SYR2K(k, n) ((k) * (n) * (n)      )

#define FADDS_TRMM_L(m, n) (0.5 * (n) * (m) * ((m)-1))
#define FMULS_TRMM_L(m, n) (0.5 * (n) * (m) * ((m)+1))

#define FADDS_TRMM_R(m, n) (0.5 * (m) * (n) * ((n)-1))
#define FMULS_TRMM_R(m, n) (0.5 * (m) * (n) * ((n)+1))

#define FADDS_TRSM_L(m, n) (0.5 * (n) * (m) * ((m)-1))
#define FMULS_TRSM_L(m, n) (0.5 * (n) * (m) * ((m)+1))

#define FADDS_TRSM_R(m, n) (0.5 * (m) * (n) * ((n)-1))
#define FMULS_TRSM_R(m, n) (0.5 * (m) * (n) * ((n)+1))

/*
 * Lapack
 */
#define FADDS_GETRF(m, n) ( ((m) < (n)) ? (0.5 * (m) * ((m) * ((n) - (1./3.) * (m)      ) - (n)) + (1. / 6.) * (m)) \
			    :             (0.5 * (n) * ((n) * ((m) - (1./3.) * (n)      ) - (m)) + (1. / 6.) * (n)) )
#define FMULS_GETRF(m, n) ( ((m) < (n)) ? (0.5 * (m) * ((m) * ((n) - (1./3.) * (m) - 1. ) + (n)) + (2. / 3.) * (m)) \
 			    :             (0.5 * (n) * ((n) * ((m) - (1./3.) * (n) - 1. ) + (m)) + (2. / 3.) * (n)) )

#define FADDS_GETRS(n, nrhs) ((nrhs) * (n) * ((n) - 1 ))
#define FMULS_GETRS(n, nrhs) ((nrhs) * (n) *  (n)      )

#define FADDS_POTRF(n) ((n) * (((1. / 6.) * (n)      ) * (n) - (1. / 6.)))
#define FMULS_POTRF(n) ((n) * (((1. / 6.) * (n) + 0.5) * (n) + (1. / 3.)))

#define FADDS_POTRS(n, nrhs) ((nrhs) * (n) * ((n) - 1 ))
#define FMULS_POTRS(n, nrhs) ((nrhs) * (n) * ((n) + 1 ))

#endif /* _FLOPS_H_ */
