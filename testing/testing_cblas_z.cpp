/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver

       @precisions normal z -> c d s
       @author Mark Gates
       
       These tests ensure that the MAGMA implementations of CBLAS routines
       are correct. (We no longer use wrappers.)
*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef HAVE_CBLAS
#include <cblas.h>
#endif

// make sure that asserts are enabled
#undef NDEBUG
#include <assert.h>

// includes, project
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
int gStatus = 0;

void output(
    const char* routine,
    int m, int n, int k, int incx, int incy,
    double error_cblas, double error_fblas )
{
    // NAN is special flag indicating not implemented -- it isn't an error
    bool okay = (isnan(error_cblas) || error_cblas < gTol) &&
                (isnan(error_fblas) || error_fblas < gTol);
    gStatus += ! okay;
    
    printf( "%5d %5d %5d %5d %5d   %-8s",
            m, n, k, incx, incy, routine );
    
    if ( isnan(error_cblas) )
        printf( "   %8s", "n/a" );
    else
        printf( "   %#8.3g", error_cblas );
    
    if ( isnan(error_fblas) )
        printf( "       %8s", "n/a" );
    else
        printf( "       %#8.3g", error_fblas );
    
    printf( "   %s\n", (okay ? "ok" : "failed") );
}


// ----------------------------------------
int main( int argc, char** argv )
{
    TESTING_INIT();
    
    //real_Double_t   t_m, t_c, t_f;
    magma_int_t ione = 1;
    
    magmaDoubleComplex  *A, *B;
    double error_cblas, error_fblas;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t m, n, k, size, maxn, ld;
    magmaDoubleComplex x2_m, x2_c, x2_f;  // complex x for magma, cblas, fortran blas respectively
    double x_m, x_c, x_f;  // x for magma, cblas, fortran blas respectively
    
    MAGMA_UNUSED( x_c  );
    MAGMA_UNUSED( x_f  );
    MAGMA_UNUSED( x2_c );
    MAGMA_UNUSED( x2_f );
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    opts.tolerance = max( 100., opts.tolerance );
    double tol = opts.tolerance * lapackf77_dlamch("E");
    gTol = tol;
    
    int inc[] = { -2, -1, 1, 2 };  //{ 1 };  //{ -1, 1 };
    int ninc = sizeof(inc)/sizeof(*inc);
    int maxinc = 0;
    for( int i=0; i < ninc; ++i ) {
        maxinc = max( maxinc, abs(inc[i]) );
    }
    
    printf( "!! Calling these CBLAS and Fortran BLAS sometimes crashes (segfaults), which !!\n"
            "!! is why we use wrappers. It does not necesarily indicate a bug in MAGMA.   !!\n"
            "!! If MAGMA_WITH_MKL or __APPLE__ are defined, known failures are skipped.   !!\n"
            "\n" );
    
    // tell user about disabled functions
    #ifndef HAVE_CBLAS
        printf( "n/a: HAVE_CBLAS not defined, so no cblas functions tested.\n\n" );
    #endif
    
    #if defined( MAGMA_WITH_MKL )
        printf( "n/a: cblas_zdotc and cblas_zdotu disabled with MKL (segfaults).\n\n" );
    #endif
    
    #if defined( __APPLE__ )
        printf( "n/a: blasf77_zdotc and blasf77_zdotu disabled on MacOS (segfaults).\n\n" );
    #endif
    
    printf( "%%                                          Error w.r.t.   Error w.r.t.\n"
            "%%   M     N     K  incx  incy   Function   CBLAS          Fortran BLAS\n"
            "%%======================================================================\n" );
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        if ( itest > 0 ) {
            printf( "%%----------------------------------------------------------------------\n" );
        }
        
        m = opts.msize[itest];
        n = opts.nsize[itest];
        k = opts.ksize[itest];
        
        // allocate matrices
        // over-allocate so they can be any combination of
        // {m,n,k} * {abs(incx), abs(incy)} by
        // {m,n,k} * {abs(incx), abs(incy)}
        maxn = max( max( m, n ), k ) * maxinc;
        ld = max( 1, maxn );
        size = ld*maxn;
        TESTING_MALLOC_CPU( A, magmaDoubleComplex, size );
        TESTING_MALLOC_CPU( B, magmaDoubleComplex, size );
        
        // initialize matrices
        lapackf77_zlarnv( &ione, ISEED, &size, A );
        lapackf77_zlarnv( &ione, ISEED, &size, B );
        
        // ----- test DZASUM
        for( int iincx = 0; iincx < ninc; ++iincx ) {
            magma_int_t incx = inc[iincx];
            
            for( int iincy = 0; iincy < ninc; ++iincy ) {
                magma_int_t incy = inc[iincy];
                
                // get one-norm of column j of A
                if ( incx > 0 && incx == incy ) {  // positive, no incy
                    error_cblas = 0;
                    error_fblas = 0;
                    for( int j = 0; j < k; ++j ) {
                        x_m = magma_cblas_dzasum( m, A(0,j), incx );
                        
                        #ifdef HAVE_CBLAS
                            x_c = cblas_dzasum( m, A(0,j), incx );
                            error_cblas = max( error_cblas, fabs( (x_m - x_c) / (m*x_c) ));
                        #else
                            error_cblas = MAGMA_D_NAN;
                        #endif
                        
                        x_f = blasf77_dzasum( &m, A(0,j), &incx );
                        error_fblas = max( error_fblas, fabs( (x_m - x_f) / (m*x_f) ));
                        
                        //printf( "xm %.8e, xc %.8e, xf %.8e\n", x_m, x_c, x_f );
                    }
                    output( "dzasum", m, n, k, incx, incy, error_cblas, error_fblas );
                }
            }
        }
        printf( "\n" );
        
        // ----- test DZNRM2
        // get two-norm of column j of A
        for( int iincx = 0; iincx < ninc; ++iincx ) {
            magma_int_t incx = inc[iincx];
            
            for( int iincy = 0; iincy < ninc; ++iincy ) {
                magma_int_t incy = inc[iincy];
                
                if ( incx > 0 && incx == incy ) {  // positive, no incy
                    error_cblas = 0;
                    error_fblas = 0;
                    for( int j = 0; j < k; ++j ) {
                        x_m = magma_cblas_dznrm2( m, A(0,j), incx );
                        
                        #ifdef HAVE_CBLAS
                            x_c = cblas_dznrm2( m, A(0,j), incx );
                            error_cblas = max( error_cblas, fabs( (x_m - x_c) / (m*x_c) ));
                        #else
                            error_cblas = MAGMA_D_NAN;
                        #endif
                        
                        x_f = blasf77_dznrm2( &m, A(0,j), &incx );
                        error_fblas = max( error_fblas, fabs( (x_m - x_f) / (m*x_f) ));
                    }
                    output( "dznrm2", m, n, k, incx, incy, error_cblas, error_fblas );
                }
            }
        }
        printf( "\n" );
        
        // ----- test ZDOTC
        // dot columns, Aj^H Bj
        for( int iincx = 0; iincx < ninc; ++iincx ) {
            magma_int_t incx = inc[iincx];
            
            for( int iincy = 0; iincy < ninc; ++iincy ) {
                magma_int_t incy = inc[iincy];
                
                error_cblas = 0;
                error_fblas = 0;
                for( int j = 0; j < k; ++j ) {
                    // MAGMA implementation, not just wrapper
                    x2_m = magma_cblas_zdotc( m, A(0,j), incx, B(0,j), incy );
                    
                    // crashes on MKL 11.1.2, ILP64
                    #if defined(HAVE_CBLAS) && ! defined( MAGMA_WITH_MKL )
                        #ifdef COMPLEX
                        cblas_zdotc_sub( m, A(0,j), incx, B(0,j), incy, &x2_c );
                        #else
                        x2_c = cblas_zdotc( m, A(0,j), incx, B(0,j), incy );
                        #endif
                        error_cblas = max( error_cblas, fabs( x2_m - x2_c ) / fabs( m*x2_c ));
                    #else
                        error_cblas = MAGMA_D_NAN;
                    #endif
                    
                    // crashes on MacOS 10.9
                    #if ! defined( __APPLE__ )
                        x2_f = blasf77_zdotc( &m, A(0,j), &incx, B(0,j), &incy );
                        error_fblas = max( error_fblas, fabs( x2_m - x2_f ) / fabs( m*x2_f ));
                    #else
                        error_fblas = MAGMA_D_NAN;
                    #endif
                        
                    //printf( "xm %.8e + %.8ei, xc %.8e + %.8ei, xf %.8e + %.8ei\n",
                    //        real(x2_m), imag(x2_m),
                    //        real(x2_c), imag(x2_c),
                    //        real(x2_f), imag(x2_f) );
                }
                output( "zdotc", m, n, k, incx, incy, error_cblas, error_fblas );
            }
        }
        printf( "\n" );
        
        // ----- test ZDOTU
        // dot columns, Aj^T * Bj
        for( int iincx = 0; iincx < ninc; ++iincx ) {
            magma_int_t incx = inc[iincx];
            
            for( int iincy = 0; iincy < ninc; ++iincy ) {
                magma_int_t incy = inc[iincy];
                
                error_cblas = 0;
                error_fblas = 0;
                for( int j = 0; j < k; ++j ) {
                    // MAGMA implementation, not just wrapper
                    x2_m = magma_cblas_zdotu( m, A(0,j), incx, B(0,j), incy );
                    
                    // crashes on MKL 11.1.2, ILP64
                    #if defined(HAVE_CBLAS) && ! defined( MAGMA_WITH_MKL )
                        #ifdef COMPLEX
                        cblas_zdotu_sub( m, A(0,j), incx, B(0,j), incy, &x2_c );
                        #else
                        x2_c = cblas_zdotu( m, A(0,j), incx, B(0,j), incy );
                        #endif
                        error_cblas = max( error_cblas, fabs( x2_m - x2_c ) / fabs( m*x2_c ));
                    #else
                        error_cblas = MAGMA_D_NAN;
                    #endif
                    
                    // crashes on MacOS 10.9
                    #if ! defined( __APPLE__ )
                        x2_f = blasf77_zdotu( &m, A(0,j), &incx, B(0,j), &incy );
                        error_fblas = max( error_fblas, fabs( x2_m - x2_f ) / fabs( m*x2_f ));
                    #else
                        error_fblas = MAGMA_D_NAN;
                    #endif
                        
                    //printf( "xm %.8e + %.8ei, xc %.8e + %.8ei, xf %.8e + %.8ei\n",
                    //        real(x2_m), imag(x2_m),
                    //        real(x2_c), imag(x2_c),
                    //        real(x2_f), imag(x2_f) );
                }
                output( "zdotu", m, n, k, incx, incy, error_cblas, error_fblas );
            }
        }
        
        // cleanup
        TESTING_FREE_CPU( A );
        TESTING_FREE_CPU( B );
        fflush( stdout );
    }  // itest, incx, incy
    
    TESTING_FINALIZE();
    return gStatus;
}
