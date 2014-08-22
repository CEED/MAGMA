/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver

       @precisions normal z -> c d s
       @author Mark Gates
       
       These tests ensure that the MAGMA wrappers around (CPU) CBLAS calls are
       correct.
       This is derived from the testing_blas_z.cpp code that checks MAGMA's
       wrappers around CUBLAS.
*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cblas.h>

// make sure that asserts are enabled
#undef NDEBUG
#include <assert.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "magma_operators.h"
#include "testings.h"

#define COMPLEX

#define A(i,j)   &A[  (i) + (j)*ld ]
#define B(i,j)   &B[  (i) + (j)*ld ]


// ----------------------------------------
// These may not be portable to different Fortran implementations,
// hence why MAGMA does not rely on them.
//#include "magma_mangling.h"

#define blasf77_dzasum FORTRAN_NAME( dzasum, DZASUM )
#define blasf77_dznrm2 FORTRAN_NAME( dznrm2, DZNRM2 )
#define blasf77_zdotc  FORTRAN_NAME( zdotc,  ZDOTC  )
#define blasf77_zdotu  FORTRAN_NAME( zdotu,  ZDOTU  )

#ifdef __cplusplus
extern "C" {
#endif

double blasf77_dzasum( const magma_int_t* n,
                       const magmaDoubleComplex* x, const magma_int_t* incx );

double blasf77_dznrm2( const magma_int_t* n,
                       const magmaDoubleComplex* x, const magma_int_t* incx );

magmaDoubleComplex blasf77_zdotc( const magma_int_t* n,
                                  const magmaDoubleComplex* x, const magma_int_t* incx,
                                  const magmaDoubleComplex* y, const magma_int_t* incy );

magmaDoubleComplex blasf77_zdotu( const magma_int_t* n,
                                  const magmaDoubleComplex* x, const magma_int_t* incx,
                                  const magmaDoubleComplex* y, const magma_int_t* incy );

#ifdef __cplusplus
}  // extern "C"
#endif


// ----------------------------------------
double gTol = 0;

const char* isok( double diff, double error )
{
    if ( diff == 0 && error < gTol ) {
        return "ok";
    }
    else {
        return "failed";
    }
}

void output( const char* routine, double diff, double error )
{
    bool ok = (diff == 0 && error < gTol);
    printf( "%-8s                                            %8.3g   %8.3g   %s\n",
            routine, diff, error, (ok ? "ok" : "failed") );
}



// ----------------------------------------
int main( int argc, char** argv )
{
    TESTING_INIT();
    
    real_Double_t   gflops, t_m, t_c, t_l;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magma_int_t ione = 1;
    
    magmaDoubleComplex  *A, *B;
    magmaDoubleComplex alpha = MAGMA_Z_MAKE( 0.5, 0.1 );
    magmaDoubleComplex beta  = MAGMA_Z_MAKE( 0.7, 0.2 );
    double work[1], diff, error;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t m, n, k, size, len, maxn, ld, info;
    magmaDoubleComplex x2_m, x2_c, x2_f;  // complex x for magma, cblas, fortran blas respectively
    double x_m, x_c, x_f, e;  // x for magma, cblas, fortran blas respectively
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    opts.tolerance = max( 100., opts.tolerance );
    double tol = opts.tolerance * lapackf77_dlamch("E");
    gTol = tol;
    
    printf( "Diff  compares MAGMA wrapper        to CBLAS and BLAS function; should be exactly 0.\n"
            "Error compares MAGMA implementation to CBLAS and BLAS function; should be ~ machine epsilon.\n"
            "\n" );
    
    double total_diff  = 0.;
    double total_error = 0.;
    int inc[] = { 1 };  //{ -2, -1, 1, 2 };  //{ 1 };  //{ -1, 1 };
    int ninc = sizeof(inc)/sizeof(*inc);
    
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        m = opts.msize[itest];
        n = opts.nsize[itest];
        k = opts.ksize[itest];
        
    for( int iincx = 0; iincx < ninc; ++iincx ) {
        magma_int_t incx = inc[iincx];
        
    for( int iincy = 0; iincy < ninc; ++iincy ) {
        magma_int_t incy = inc[iincy];
        
        printf("=========================================================================\n");
        printf( "m=%d, n=%d, k=%d, incx = %d, incy = %d\n",
                (int) m, (int) n, (int) k, (int) incx, (int) incy );
        printf( "Function              MAGMA     CBLAS     BLAS        Diff      Error\n"
                "                      msec      msec      msec\n" );
        
        // allocate matrices
        // over-allocate so they can be any combination of
        // {m,n,k} * {abs(incx), abs(incy)} by
        // {m,n,k} * {abs(incx), abs(incy)}
        maxn = max( max( m, n ), k ) * max( abs(incx), abs(incy) );
        ld = max( 1, maxn );
        size = ld*maxn;
        magma_zmalloc_pinned( &A,  size );  assert( A   != NULL );
        magma_zmalloc_pinned( &B,  size );  assert( B   != NULL );
        
        // initialize matrices
        lapackf77_zlarnv( &ione, ISEED, &size, A );
        lapackf77_zlarnv( &ione, ISEED, &size, B );
        
        printf( "Level 1 BLAS ----------------------------------------------------------\n" );
        
        
        // ----- test DZASUM
        // get one-norm of column j of A
        if ( incx > 0 && incx == incy ) {  // positive, no incy
            diff  = 0;
            error = 0;
            for( int j = 0; j < k; ++j ) {
                x_m = magma_cblas_dzasum( m, A(0,j), incx );
                
                x_c = cblas_dzasum( m, A(0,j), incx );
                diff += fabs( x_m - x_c );
                
                x_f = blasf77_dzasum( &m, A(0,j), &incx );
                diff += fabs( x_m - x_f );
            }
            output( "dzasum", diff, error );
            total_diff  += diff;
            total_error += error;
        }
        
        // ----- test DZNRM2
        // get two-norm of column j of A
        if ( incx > 0 && incx == incy ) {  // positive, no incy
            diff  = 0;
            error = 0;
            for( int j = 0; j < k; ++j ) {
                x_m = magma_cblas_dznrm2( m, A(0,j), incx );
                
                x_c = cblas_dznrm2( m, A(0,j), incx );
                diff += fabs( x_m - x_c );
                
                x_f = blasf77_dznrm2( &m, A(0,j), &incx );
                diff += fabs( x_m - x_f );
            }
            output( "dznrm2", diff, error );
            total_diff  += diff;
            total_error += error;
        }
        
        // ----- test ZDOTC
        // dot columns, Aj^H Bj
        diff  = 0;
        error = 0;
        for( int j = 0; j < k; ++j ) {
            // MAGMA implementation, not just wrapper
            x2_m = magma_cblas_zdotc( m, A(0,j), incx, B(0,j), incy );
            
            #ifdef COMPLEX
            cblas_zdotc_sub( m, A(0,j), incx, B(0,j), incy, &x2_c );
            #else
            x2_c = cblas_zdotc( m, A(0,j), incx, B(0,j), incy );
            #endif
            error += fabs( x2_m - x2_c ) / fabs( m*x2_c );
            
            // crashes (on MacOS)
            //x2_f = blasf77_zdotc( &m, A(0,j), &incx, B(0,j), &incy );
            //error += fabs( x2_m - x2_f ) / fabs( m*x2_f );
        }
        output( "zdotc", diff, error );
        total_diff  += diff;
        total_error += error;
        total_error += error;
        
        // ----- test ZDOTU
        // dot columns, Aj^T * Bj
        diff  = 0;
        error = 0;
        for( int j = 0; j < k; ++j ) {
            // MAGMA implementation, not just wrapper
            x2_m = magma_cblas_zdotu( m, A(0,j), incx, B(0,j), incy );
            
            #ifdef COMPLEX
            cblas_zdotu_sub( m, A(0,j), incx, B(0,j), incy, &x2_c );
            #else
            x2_c = cblas_zdotu( m, A(0,j), incx, B(0,j), incy );
            #endif
            error += fabs( x2_m - x2_c ) / fabs( m*x2_c );
            
            // crashes (on MacOS)
            //x2_f = blasf77_zdotu( &m, A(0,j), &incx, B(0,j), &incy );
            //error += fabs( x2_m - x2_f ) / fabs( m*x2_f );
        }
        output( "zdotu", diff, error );
        total_diff  += diff;
        total_error += error;
        
        // cleanup
        magma_free_pinned( A );
        magma_free_pinned( B );
        fflush( stdout );
    }}}  // itest, incx, incy
    
    // TODO use average error?
    printf( "sum diffs  = %8.2g, MAGMA wrapper        compared to CBLAS and Fortran BLAS; should be exactly 0.\n"
            "sum errors = %8.2e, MAGMA implementation compared to CBLAS and Fortran BLAS; should be ~ machine epsilon.\n\n",
            total_diff, total_error );
    if ( total_diff != 0. ) {
        printf( "some tests failed diff == 0.; see above.\n" );
    }
    else {
        printf( "all tests passed diff == 0.\n" );
    }
    
    TESTING_FINALIZE();
    
    int status = (total_diff != 0.);
    return status;
}
