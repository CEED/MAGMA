/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
*/

#ifndef MAGMA_TYPES_H
#define MAGMA_TYPES_H

#include <stdint.h>
#include <assert.h>


// each implementation of MAGMA defines HAVE_* appropriately.
#if ! defined(HAVE_CUBLAS) && ! defined(HAVE_clBLAS) && ! defined(HAVE_MIC)
#define HAVE_CUBLAS
#endif


// ========================================
// C99 standard defines __func__. Some older compilers use __FUNCTION__.
// Note __func__ in C99 is not a macro, so ifndef __func__ doesn't work.
#if __STDC_VERSION__ < 199901L
  #ifndef __func__
    #if __GNUC__ >= 2 || _MSC_VER >= 1300
      #define __func__ __FUNCTION__
    #else
      #define __func__ "<unknown>"
    #endif
  #endif
#endif


// ========================================
// To use int64_t, link with mkl_intel_ilp64 or similar (instead of mkl_intel_lp64).
// Similar to magma_int_t we declare magma_index_t used for row/column indices in sparse
#if defined(MAGMA_ILP64) || defined(MKL_ILP64)
//typedef int64_t magma_int_t;
typedef long long int magma_int_t;  // MKL uses long long int, not int64_t
#else
typedef int magma_int_t;
#endif

typedef int magma_index_t;

// Define new type that the precision generator will not change (matches PLASMA)
typedef double real_Double_t;


// ========================================
// opaque queue structure
struct magma_queue;
typedef struct magma_queue* magma_queue_t;



// ========================================
// define types specific to implementation (CUDA, OpenCL, MIC)
// define macros to deal with complex numbers
#if defined(HAVE_CUBLAS)
    // include cublas_v2.h, unless cublas.h has already been included, e.g., via magma.h
    #ifndef CUBLAS_H_
    #include <cublas_v2.h>
    #endif
    
    #include <cusparse_v2.h>
    
    #ifdef __cplusplus
    extern "C" {
    #endif
    
    magma_int_t      magma_queue_get_device         ( magma_queue_t queue );
    cudaStream_t     magma_queue_get_cuda_stream    ( magma_queue_t queue );
    cublasHandle_t   magma_queue_get_cublas_handle  ( magma_queue_t queue );
    cusparseHandle_t magma_queue_get_cusparse_handle( magma_queue_t queue );
    
    #ifdef __cplusplus
    }
    #endif
    
    typedef cudaEvent_t    magma_event_t;
    typedef int            magma_device_t;
    
    typedef cuDoubleComplex magmaDoubleComplex;
    typedef cuFloatComplex  magmaFloatComplex;
    
    #define MAGMA_Z_MAKE(r,i)     make_cuDoubleComplex(r, i)
    #define MAGMA_Z_REAL(a)       (a).x
    #define MAGMA_Z_IMAG(a)       (a).y
    #define MAGMA_Z_ADD(a, b)     cuCadd(a, b)
    #define MAGMA_Z_SUB(a, b)     cuCsub(a, b)
    #define MAGMA_Z_MUL(a, b)     cuCmul(a, b)
    #define MAGMA_Z_DIV(a, b)     cuCdiv(a, b)
    #define MAGMA_Z_ABS(a)        cuCabs(a)
    #define MAGMA_Z_ABS1(a)       (fabs((a).x) + fabs((a).y))
    #define MAGMA_Z_CONJ(a)       cuConj(a)
    
    #define MAGMA_C_MAKE(r,i)     make_cuFloatComplex(r, i)
    #define MAGMA_C_REAL(a)       (a).x
    #define MAGMA_C_IMAG(a)       (a).y
    #define MAGMA_C_ADD(a, b)     cuCaddf(a, b)
    #define MAGMA_C_SUB(a, b)     cuCsubf(a, b)
    #define MAGMA_C_MUL(a, b)     cuCmulf(a, b)
    #define MAGMA_C_DIV(a, b)     cuCdivf(a, b)
    #define MAGMA_C_ABS(a)        cuCabsf(a)
    #define MAGMA_C_ABS1(a)       (fabsf((a).x) + fabsf((a).y))
    #define MAGMA_C_CONJ(a)       cuConjf(a)
    
#elif defined(HAVE_clBLAS)
    #include <clBLAS.h>
    
    typedef cl_event          magma_event_t;
    typedef cl_device_id      magma_device_t;
    
    typedef DoubleComplex magmaDoubleComplex;
    typedef FloatComplex  magmaFloatComplex;
    
    #define MAGMA_Z_MAKE(r,i)     doubleComplex(r,i)
    #define MAGMA_Z_REAL(a)       (a).x
    #define MAGMA_Z_IMAG(a)       (a).y
    #define MAGMA_Z_ADD(a, b)     MAGMA_Z_MAKE((a).x+(b).x, (a).y+(b).y)
    #define MAGMA_Z_SUB(a, b)     MAGMA_Z_MAKE((a).x-(b).x, (a).y-(b).y)
    #define MAGMA_Z_DIV(a, b)     ((a)/(b))
    #define MAGMA_Z_ABS(a)        magma_cabs(a)
    #define MAGMA_Z_ABS1(a)       (fabs((a).x) + fabs((a).y))
    #define MAGMA_Z_CONJ(a)       MAGMA_Z_MAKE((a).x, -(a).y)
    
    #define MAGMA_C_MAKE(r,i)     floatComplex(r,i)
    #define MAGMA_C_REAL(a)       (a).x
    #define MAGMA_C_IMAG(a)       (a).y
    #define MAGMA_C_ADD(a, b)     MAGMA_C_MAKE((a).x+(b).x, (a).y+(b).y)
    #define MAGMA_C_SUB(a, b)     MAGMA_C_MAKE((a).x-(b).x, (a).y-(b).y)
    #define MAGMA_C_DIV(a, b)     ((a)/(b))
    #define MAGMA_C_ABS(a)        magma_cabsf(a)
    #define MAGMA_C_ABS1(a)       (fabsf((a).x) + fabsf((a).y))
    #define MAGMA_C_CONJ(a)       MAGMA_C_MAKE((a).x, -(a).y)

#elif defined(HAVE_MIC)
    #include <stdio.h>
    #include <stdlib.h>
    #include <stdint.h>
    #include <unistd.h>
    #include <fcntl.h>
    #include <string.h>
    #include <sys/mman.h>
    #include <sys/ioctl.h>
    #include <sys/time.h>
    #include <scif.h>
    //#include <mkl.h>

    typedef int   magma_event_t;
    typedef int   magma_device_t;

    #include <complex>
    typedef std::complex<float>   magmaFloatComplex;
    typedef std::complex<double>  magmaDoubleComplex;

    #define MAGMA_Z_MAKE(r, i)    std::complex<double>(r,i)
    #define MAGMA_Z_REAL(x)       (x).real()
    #define MAGMA_Z_IMAG(x)       (x).imag()
    #define MAGMA_Z_ADD(a, b)     ((a)+(b))
    #define MAGMA_Z_SUB(a, b)     ((a)-(b))
    #define MAGMA_Z_MUL(a, b)     ((a)*(b))
    #define MAGMA_Z_DIV(a, b)     ((a)/(b))
    #define MAGMA_Z_ABS(a)        abs(a)
    #define MAGMA_Z_ABS1(a)       (fabs((a).real()) + fabs((a).imag())) 
    #define MAGMA_Z_CONJ(a)       conj(a)

    #define MAGMA_C_MAKE(r, i)    std::complex<float> (r,i)
    #define MAGMA_C_REAL(x)       (x).real()
    #define MAGMA_C_IMAG(x)       (x).imag()
    #define MAGMA_C_ADD(a, b)     ((a)+(b))
    #define MAGMA_C_SUB(a, b)     ((a)-(b))
    #define MAGMA_C_MUL(a, b)     ((a)*(b))
    #define MAGMA_C_DIV(a, b)     ((a)/(b))
    #define MAGMA_C_ABS(a)        abs(a)
    #define MAGMA_C_ABS1(a)       (fabs((a).real()) + fabs((a).imag()))
    #define MAGMA_C_CONJ(a)       conj(a)
#else
    #error "One of HAVE_CUBLAS, HAVE_clBLAS, or HAVE_MIC must be defined. For example, add -DHAVE_CUBLAS to CFLAGS, or #define HAVE_CUBLAS before #include <magma.h>. In MAGMA, this happens in Makefile.internal."
#endif

#define MAGMA_Z_EQUAL(a,b)        (MAGMA_Z_REAL(a)==MAGMA_Z_REAL(b) && MAGMA_Z_IMAG(a)==MAGMA_Z_IMAG(b))
#define MAGMA_Z_NEGATE(a)         MAGMA_Z_MAKE( -MAGMA_Z_REAL(a), -MAGMA_Z_IMAG(a))

#define MAGMA_C_EQUAL(a,b)        (MAGMA_C_REAL(a)==MAGMA_C_REAL(b) && MAGMA_C_IMAG(a)==MAGMA_C_IMAG(b))
#define MAGMA_C_NEGATE(a)         MAGMA_C_MAKE( -MAGMA_C_REAL(a), -MAGMA_C_IMAG(a))

#define MAGMA_D_MAKE(r,i)         (r)
#define MAGMA_D_REAL(x)           (x)
#define MAGMA_D_IMAG(x)           (0.0)
#define MAGMA_D_ADD(a, b)         ((a) + (b))
#define MAGMA_D_SUB(a, b)         ((a) - (b))
#define MAGMA_D_MUL(a, b)         ((a) * (b))
#define MAGMA_D_DIV(a, b)         ((a) / (b))
#define MAGMA_D_ABS(a)            ((a)>0 ? (a) : -(a))
#define MAGMA_D_ABS1(a)           ((a)>0 ? (a) : -(a))
#define MAGMA_D_CONJ(a)           (a)
#define MAGMA_D_EQUAL(a,b)        ((a) == (b))
#define MAGMA_D_NEGATE(a)         (-a)

#define MAGMA_S_MAKE(r,i)         (r)
#define MAGMA_S_REAL(x)           (x)
#define MAGMA_S_IMAG(x)           (0.0)
#define MAGMA_S_ADD(a, b)         ((a) + (b))
#define MAGMA_S_SUB(a, b)         ((a) - (b))
#define MAGMA_S_MUL(a, b)         ((a) * (b))
#define MAGMA_S_DIV(a, b)         ((a) / (b))
#define MAGMA_S_ABS(a)            ((a)>0 ? (a) : -(a))
#define MAGMA_S_ABS1(a)           ((a)>0 ? (a) : -(a))
#define MAGMA_S_CONJ(a)           (a)
#define MAGMA_S_EQUAL(a,b)        ((a) == (b))
#define MAGMA_S_NEGATE(a)         (-a)

#define MAGMA_Z_ZERO              MAGMA_Z_MAKE( 0.0, 0.0)
#define MAGMA_Z_ONE               MAGMA_Z_MAKE( 1.0, 0.0)
#define MAGMA_Z_HALF              MAGMA_Z_MAKE( 0.5, 0.0)
#define MAGMA_Z_NEG_ONE           MAGMA_Z_MAKE(-1.0, 0.0)
#define MAGMA_Z_NEG_HALF          MAGMA_Z_MAKE(-0.5, 0.0)

#define MAGMA_C_ZERO              MAGMA_C_MAKE( 0.0, 0.0)
#define MAGMA_C_ONE               MAGMA_C_MAKE( 1.0, 0.0)
#define MAGMA_C_HALF              MAGMA_C_MAKE( 0.5, 0.0)
#define MAGMA_C_NEG_ONE           MAGMA_C_MAKE(-1.0, 0.0)
#define MAGMA_C_NEG_HALF          MAGMA_C_MAKE(-0.5, 0.0)

#define MAGMA_D_ZERO              ( 0.0)
#define MAGMA_D_ONE               ( 1.0)
#define MAGMA_D_HALF              ( 0.5)
#define MAGMA_D_NEG_ONE           (-1.0)
#define MAGMA_D_NEG_HALF          (-0.5)

#define MAGMA_S_ZERO              ( 0.0)
#define MAGMA_S_ONE               ( 1.0)
#define MAGMA_S_HALF              ( 0.5)
#define MAGMA_S_NEG_ONE           (-1.0)
#define MAGMA_S_NEG_HALF          (-0.5)

#ifndef CBLAS_SADDR
#define CBLAS_SADDR(a)  &(a)
#endif

#if defined(HAVE_clBLAS)
    // OpenCL uses opaque memory references on GPU
    typedef cl_mem magma_ptr;
    typedef cl_mem magmaInt_ptr;
    typedef cl_mem magmaIndex_ptr;
    typedef cl_mem magmaFloat_ptr;
    typedef cl_mem magmaDouble_ptr;
    typedef cl_mem magmaFloatComplex_ptr;
    typedef cl_mem magmaDoubleComplex_ptr;
    
    typedef cl_mem magma_const_ptr;
    typedef cl_mem magmaInt_const_ptr;
    typedef cl_mem magmaIndex_const_ptr;
    typedef cl_mem magmaFloat_const_ptr;
    typedef cl_mem magmaDouble_const_ptr;
    typedef cl_mem magmaFloatComplex_const_ptr;
    typedef cl_mem magmaDoubleComplex_const_ptr;
#else
    // MIC and CUDA use regular pointers on GPU
    typedef void               *magma_ptr;
    typedef magma_int_t        *magmaInt_ptr;
    typedef magma_index_t      *magmaIndex_ptr;
    typedef float              *magmaFloat_ptr;
    typedef double             *magmaDouble_ptr;
    typedef magmaFloatComplex  *magmaFloatComplex_ptr;
    typedef magmaDoubleComplex *magmaDoubleComplex_ptr;
    
    typedef void               const *magma_const_ptr;
    typedef magma_int_t        const *magmaInt_const_ptr;
    typedef magma_index_t      const *magmaIndex_const_ptr;
    typedef float              const *magmaFloat_const_ptr;
    typedef double             const *magmaDouble_const_ptr;
    typedef magmaFloatComplex  const *magmaFloatComplex_const_ptr;
    typedef magmaDoubleComplex const *magmaDoubleComplex_const_ptr;
#endif


// ========================================
// MAGMA constants

// ----------------------------------------
#define MAGMA_VERSION_MAJOR 1
#define MAGMA_VERSION_MINOR 7
#define MAGMA_VERSION_MICRO 0

// stage is "svn", "beta#", "rc#" (release candidate), or blank ("") for final release
#define MAGMA_VERSION_STAGE "svn"

#define MagmaMaxGPUs 8
#define MagmaMaxAccelerators 8


// ----------------------------------------
// Return codes
// LAPACK argument errors are < 0 but > MAGMA_ERR.
// MAGMA errors are < MAGMA_ERR.
#define MAGMA_SUCCESS               0
#define MAGMA_ERR                  -100
#define MAGMA_ERR_NOT_INITIALIZED  -101
#define MAGMA_ERR_REINITIALIZED    -102
#define MAGMA_ERR_NOT_SUPPORTED    -103
#define MAGMA_ERR_ILLEGAL_VALUE    -104
#define MAGMA_ERR_NOT_FOUND        -105
#define MAGMA_ERR_ALLOCATION       -106
#define MAGMA_ERR_INTERNAL_LIMIT   -107
#define MAGMA_ERR_UNALLOCATED      -108
#define MAGMA_ERR_FILESYSTEM       -109
#define MAGMA_ERR_UNEXPECTED       -110
#define MAGMA_ERR_SEQUENCE_FLUSHED -111
#define MAGMA_ERR_HOST_ALLOC       -112
#define MAGMA_ERR_DEVICE_ALLOC     -113
#define MAGMA_ERR_CUDASTREAM       -114
#define MAGMA_ERR_INVALID_PTR      -115
#define MAGMA_ERR_UNKNOWN          -116
#define MAGMA_ERR_NOT_IMPLEMENTED  -117
#define MAGMA_ERR_NAN              -118

// some sparse-iter errors
#define MAGMA_SLOW_CONVERGENCE     -201
#define MAGMA_DIVERGENCE           -202
#define MAGMA_NONSPD               -203
#define MAGMA_ERR_BADPRECOND       -204
#define MAGMA_NOTCONVERGED         -205

// When adding error codes, please add to interface_cuda/error.cpp

// map cusparse errors to magma errors
#define MAGMA_ERR_CUSPARSE_NOT_INITIALIZED            -3001
#define MAGMA_ERR_CUSPARSE_ALLOC_FAILED               -3002
#define MAGMA_ERR_CUSPARSE_INVALID_VALUE              -3003
#define MAGMA_ERR_CUSPARSE_ARCH_MISMATCH              -3004
#define MAGMA_ERR_CUSPARSE_MAPPING_ERROR              -3005
#define MAGMA_ERR_CUSPARSE_EXECUTION_FAILED           -3006
#define MAGMA_ERR_CUSPARSE_INTERNAL_ERROR             -3007
#define MAGMA_ERR_CUSPARSE_MATRIX_TYPE_NOT_SUPPORTED  -3008
#define MAGMA_ERR_CUSPARSE_ZERO_PIVOT                 -3009

// some template parameter
#define MagmaBigTileSize 1000000    

// ----------------------------------------
// parameter constants
// numbering is consistent with CBLAS and PLASMA; see plasma/include/plasma.h
// also with lapack_cwrapper/include/lapack_enum.h
typedef enum {
    MagmaFalse         = 0,
    MagmaTrue          = 1
} magma_bool_t;

typedef enum {
    MagmaRowMajor      = 101,
    MagmaColMajor      = 102
} magma_order_t;

// Magma_ConjTrans is an alias for those rare occasions (zlarfb, zun*, zher*k)
// where we want Magma_ConjTrans to convert to MagmaTrans in precision generation.
typedef enum {
    MagmaNoTrans       = 111,
    MagmaTrans         = 112,
    MagmaConjTrans     = 113,
    Magma_ConjTrans    = MagmaConjTrans
} magma_trans_t;

typedef enum {
    MagmaUpper         = 121,
    MagmaLower         = 122,
    MagmaFull          = 123,  /* lascl, laset */
    MagmaHessenberg    = 124   /* lascl */
} magma_uplo_t;

typedef magma_uplo_t magma_type_t;  /* lascl */

typedef enum {
    MagmaNonUnit       = 131,
    MagmaUnit          = 132
} magma_diag_t;

typedef enum {
    MagmaLeft          = 141,
    MagmaRight         = 142,
    MagmaBothSides     = 143   /* trevc */
} magma_side_t;

typedef enum {
    MagmaOneNorm       = 171,  /* lange, lanhe */
    MagmaRealOneNorm   = 172,
    MagmaTwoNorm       = 173,
    MagmaFrobeniusNorm = 174,
    MagmaInfNorm       = 175,
    MagmaRealInfNorm   = 176,
    MagmaMaxNorm       = 177,
    MagmaRealMaxNorm   = 178
} magma_norm_t;

typedef enum {
    MagmaDistUniform   = 201,  /* latms */
    MagmaDistSymmetric = 202,
    MagmaDistNormal    = 203
} magma_dist_t;

typedef enum {
    MagmaHermGeev      = 241,  /* latms */
    MagmaHermPoev      = 242,
    MagmaNonsymPosv    = 243,
    MagmaSymPosv       = 244
} magma_sym_t;

typedef enum {
    MagmaNoPacking     = 291,  /* latms */
    MagmaPackSubdiag   = 292,
    MagmaPackSupdiag   = 293,
    MagmaPackColumn    = 294,
    MagmaPackRow       = 295,
    MagmaPackLowerBand = 296,
    MagmaPackUpeprBand = 297,
    MagmaPackAll       = 298
} magma_pack_t;

typedef enum {
    MagmaNoVec         = 301,  /* geev, syev, gesvd */
    MagmaVec           = 302,  /* geev, syev */
    MagmaIVec          = 303,  /* stedc */
    MagmaAllVec        = 304,  /* gesvd, trevc */
    MagmaSomeVec       = 305,  /* gesvd, trevc */
    MagmaOverwriteVec  = 306,  /* gesvd */
    MagmaBacktransVec  = 307   /* trevc */
} magma_vec_t;

typedef enum {
    MagmaRangeAll      = 311,  /* syevx, etc. */
    MagmaRangeV        = 312,
    MagmaRangeI        = 313
} magma_range_t;

typedef enum {
    MagmaQ             = 322,  /* unmbr, ungbr */
    MagmaP             = 323
} magma_vect_t;

typedef enum {
    MagmaForward       = 391,  /* larfb */
    MagmaBackward      = 392
} magma_direct_t;

typedef enum {
    MagmaColumnwise    = 401,  /* larfb */
    MagmaRowwise       = 402
} magma_storev_t;

// --------------------
// sparse
typedef enum {
    Magma_CSR          = 611,
    Magma_ELLPACKT     = 612,
    Magma_ELL          = 613,
    Magma_DENSE        = 614,
    Magma_BCSR         = 615,
    Magma_CSC          = 616,
    Magma_HYB          = 617,
    Magma_COO          = 618,
    Magma_ELLRT        = 619,
    Magma_SPMVFUNCTION = 620,
    Magma_SELLP        = 621,
    Magma_ELLD         = 622,
    Magma_CSRLIST      = 623,
    Magma_CSRD         = 624,
    Magma_CSRL         = 627,
    Magma_CSRU         = 628,
    Magma_CSRCOO       = 629,
    Magma_CUCSR        = 630
} magma_storage_t;


typedef enum {
    Magma_CG           = 431,
    Magma_CGMERGE      = 432,
    Magma_GMRES        = 433,
    Magma_BICGSTAB     = 434,
  Magma_BICGSTABMERGE  = 435,
  Magma_BICGSTABMERGE2 = 436,
    Magma_JACOBI       = 437,
    Magma_GS           = 438,
    Magma_ITERREF      = 439,
    Magma_BCSRLU       = 440,
    Magma_PCG          = 441,
    Magma_PGMRES       = 442,
    Magma_PBICGSTAB    = 443,
    Magma_PASTIX       = 444,
    Magma_ILU          = 445,
    Magma_ICC          = 446,
    Magma_AILU         = 447,
    Magma_AICC         = 448,
    Magma_BAITER       = 449,
    Magma_LOBPCG       = 450,
    Magma_NONE         = 451,
    Magma_FUNCTION     = 452,
    Magma_IDR          = 453,
    Magma_PIDR         = 454,
    Magma_CGS          = 455,
    Magma_PCGS         = 456,
    Magma_CGSMERGE     = 457,
    Magma_PCGSMERGE    = 458,
    Magma_TFQMR        = 459,   
    Magma_PTFQMR       = 460,
    Magma_TFQMRMERGE   = 461,   
    Magma_PTFQMRMERGE  = 462,
    Magma_QMR          = 463,   
    Magma_PQMR         = 464,
    Magma_QMRMERGE     = 465,   
    Magma_PQMRMERGE    = 466,
    Magma_BOMBARD      = 490,
    Magma_BOMBARDMERGE = 491,
    Magma_PCGMERGE     = 492,
    Magma_BAITERO      = 493,
    Magma_IDRMERGE     = 494,
  Magma_PBICGSTABMERGE = 495,
    Magma_AICT         = 496,
    Magma_CUSTOMIC     = 497,
    Magma_CUSTOMILU    = 498,
    Magma_PIDRMERGE    = 499,
    Magma_BICG         = 500,
    Magma_BICGMERGE    = 501,
    Magma_PBICG        = 502,
    Magma_PBICGMERGE   = 503
    
} magma_solver_type;

typedef enum {
    Magma_CGSO         = 561,
    Magma_FUSED_CGSO   = 562,
    Magma_MGSO         = 563
} magma_ortho_t;

typedef enum {
    Magma_CPU          = 571,
    Magma_DEV          = 572
} magma_location_t;

typedef enum {
    Magma_GENERAL      = 581,
    Magma_SYMMETRIC    = 582
} magma_symmetry_t;

typedef enum {
    Magma_ORDERED      = 591,
    Magma_DIAGFIRST    = 592,
    Magma_UNITY        = 593,
    Magma_VALUE        = 594
} magma_diagorder_t;

typedef enum {
    Magma_DCOMPLEX     = 501,
    Magma_FCOMPLEX     = 502,
    Magma_DOUBLE       = 503,
    Magma_FLOAT        = 504
} magma_precision;

typedef enum {
    Magma_NOSCALE      = 511,
    Magma_UNITROW      = 512,
    Magma_UNITDIAG     = 513
} magma_scale_t;


// When adding constants, remember to do these steps as appropriate:
// 1)  add magma_xxxx_const()  converter below and in control/constants.cpp
// 2a) add to magma2lapack_constants[] in control/constants.cpp
// 2b) update min & max here, which are used to check bounds for magma2lapack_constants[]
// 2c) add lapack_xxxx_const() converter below and in control/constants.cpp
#define Magma2lapack_Min  MagmaFalse     // 0
#define Magma2lapack_Max  MagmaRowwise   // 402


// ----------------------------------------
// string constants for calling Fortran BLAS and LAPACK
// todo: use translators instead? lapack_const_str( MagmaUpper )
#define MagmaRowMajorStr      "Row"
#define MagmaColMajorStr      "Col"

#define MagmaNoTransStr       "NoTrans"
#define MagmaTransStr         "Trans"
#define MagmaConjTransStr     "ConjTrans"
#define Magma_ConjTransStr    "ConjTrans"

#define MagmaUpperStr         "Upper"
#define MagmaLowerStr         "Lower"
#define MagmaFullStr          "Full"

#define MagmaNonUnitStr       "NonUnit"
#define MagmaUnitStr          "Unit"

#define MagmaLeftStr          "Left"
#define MagmaRightStr         "Right"
#define MagmaBothSidesStr     "Both"

#define MagmaOneNormStr       "1"
#define MagmaTwoNormStr       "2"
#define MagmaFrobeniusNormStr "Fro"
#define MagmaInfNormStr       "Inf"
#define MagmaMaxNormStr       "Max"

#define MagmaForwardStr       "Forward"
#define MagmaBackwardStr      "Backward"

#define MagmaColumnwiseStr    "Columnwise"
#define MagmaRowwiseStr       "Rowwise"

#define MagmaNoVecStr         "NoVec"
#define MagmaVecStr           "Vec"
#define MagmaIVecStr          "IVec"
#define MagmaAllVecStr        "All"
#define MagmaSomeVecStr       "Some"
#define MagmaOverwriteVecStr  "Overwrite"


#ifdef __cplusplus
extern "C" {
#endif

// --------------------
// Convert LAPACK character constants to MAGMA constants.
// This is a one-to-many mapping, requiring multiple translators
// (e.g., "N" can be NoTrans or NonUnit or NoVec).
magma_bool_t   magma_bool_const  ( char lapack_char );
magma_order_t  magma_order_const ( char lapack_char );
magma_trans_t  magma_trans_const ( char lapack_char );
magma_uplo_t   magma_uplo_const  ( char lapack_char );
magma_diag_t   magma_diag_const  ( char lapack_char );
magma_side_t   magma_side_const  ( char lapack_char );
magma_norm_t   magma_norm_const  ( char lapack_char );
magma_dist_t   magma_dist_const  ( char lapack_char );
magma_sym_t    magma_sym_const   ( char lapack_char );
magma_pack_t   magma_pack_const  ( char lapack_char );
magma_vec_t    magma_vec_const   ( char lapack_char );
magma_range_t  magma_range_const ( char lapack_char );
magma_vect_t   magma_vect_const  ( char lapack_char );
magma_direct_t magma_direct_const( char lapack_char );
magma_storev_t magma_storev_const( char lapack_char );


// --------------------
// Convert MAGMA constants to LAPACK(E) constants.
// The generic lapack_const_str works for all cases, but the specific routines
// (e.g., lapack_trans_const) do better error checking.

// magma  defines lapack_const_str, which returns char* to call lapack (Fortran interface).
// plasma defines lapack_const, which is roughly the same as MAGMA's lapacke_const
// (returns a char instead of char*) to call lapacke (C interface).

const char* lapack_const_str   ( int            magma_const );
const char* lapack_bool_const  ( magma_bool_t   magma_const );
const char* lapack_order_const ( magma_order_t  magma_const );
const char* lapack_trans_const ( magma_trans_t  magma_const );
const char* lapack_uplo_const  ( magma_uplo_t   magma_const );
const char* lapack_diag_const  ( magma_diag_t   magma_const );
const char* lapack_side_const  ( magma_side_t   magma_const );
const char* lapack_norm_const  ( magma_norm_t   magma_const );
const char* lapack_dist_const  ( magma_dist_t   magma_const );
const char* lapack_sym_const   ( magma_sym_t    magma_const );
const char* lapack_pack_const  ( magma_pack_t   magma_const );
const char* lapack_vec_const   ( magma_vec_t    magma_const );
const char* lapack_range_const ( magma_range_t  magma_const );
const char* lapack_vect_const  ( magma_vect_t   magma_const );
const char* lapack_direct_const( magma_direct_t magma_const );
const char* lapack_storev_const( magma_storev_t magma_const );

static inline char lapacke_const       ( int magma_const            ) { return *lapack_const_str   ( magma_const ); }
static inline char lapacke_bool_const  ( magma_bool_t   magma_const ) { return *lapack_bool_const  ( magma_const ); }
static inline char lapacke_order_const ( magma_order_t  magma_const ) { return *lapack_order_const ( magma_const ); }
static inline char lapacke_trans_const ( magma_trans_t  magma_const ) { return *lapack_trans_const ( magma_const ); }
static inline char lapacke_uplo_const  ( magma_uplo_t   magma_const ) { return *lapack_uplo_const  ( magma_const ); }
static inline char lapacke_diag_const  ( magma_diag_t   magma_const ) { return *lapack_diag_const  ( magma_const ); }
static inline char lapacke_side_const  ( magma_side_t   magma_const ) { return *lapack_side_const  ( magma_const ); }
static inline char lapacke_norm_const  ( magma_norm_t   magma_const ) { return *lapack_norm_const  ( magma_const ); }
static inline char lapacke_dist_const  ( magma_dist_t   magma_const ) { return *lapack_dist_const  ( magma_const ); }
static inline char lapacke_sym_const   ( magma_sym_t    magma_const ) { return *lapack_sym_const   ( magma_const ); }
static inline char lapacke_pack_const  ( magma_pack_t   magma_const ) { return *lapack_pack_const  ( magma_const ); }
static inline char lapacke_vec_const   ( magma_vec_t    magma_const ) { return *lapack_vec_const   ( magma_const ); }
static inline char lapacke_range_const ( magma_range_t  magma_const ) { return *lapack_range_const ( magma_const ); }
static inline char lapacke_vect_const  ( magma_vect_t   magma_const ) { return *lapack_vect_const  ( magma_const ); }
static inline char lapacke_direct_const( magma_direct_t magma_const ) { return *lapack_direct_const( magma_const ); }
static inline char lapacke_storev_const( magma_storev_t magma_const ) { return *lapack_storev_const( magma_const ); }


// --------------------
// Convert MAGMA constants to clBLAS constants.
#if defined(HAVE_clBLAS)
clblasOrder          clblas_order_const( magma_order_t order );
clblasTranspose      clblas_trans_const( magma_trans_t trans );
clblasUplo           clblas_uplo_const ( magma_uplo_t  uplo  );
clblasDiag           clblas_diag_const ( magma_diag_t  diag  );
clblasSide           clblas_side_const ( magma_side_t  side  );
#endif


// --------------------
// Convert MAGMA constants to CUBLAS constants.
#if defined(CUBLAS_V2_H_)
cublasOperation_t    cublas_trans_const ( magma_trans_t trans );
cublasFillMode_t     cublas_uplo_const  ( magma_uplo_t  uplo  );
cublasDiagType_t     cublas_diag_const  ( magma_diag_t  diag  );
cublasSideMode_t     cublas_side_const  ( magma_side_t  side  );
#endif


// --------------------
// Convert MAGMA constants to CBLAS constants.
#if defined(HAVE_CBLAS)
#include <cblas.h>
enum CBLAS_ORDER     cblas_order_const  ( magma_order_t order );
enum CBLAS_TRANSPOSE cblas_trans_const  ( magma_trans_t trans );
enum CBLAS_UPLO      cblas_uplo_const   ( magma_uplo_t  uplo  );
enum CBLAS_DIAG      cblas_diag_const   ( magma_diag_t  diag  );
enum CBLAS_SIDE      cblas_side_const   ( magma_side_t  side  );
#endif


#ifdef __cplusplus
}
#endif

#endif        //  #ifndef MAGMA_TYPES_H
