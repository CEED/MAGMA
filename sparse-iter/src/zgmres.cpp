/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @author Stan Tomov
       @author Hartwig Anzt

       @precisions normal z -> s d c
*/
#include <sys/time.h>
#include <time.h>

#include "common_magma.h"
#include "../include/magmasparse.h"
#include "trace.h"
extern "C" magma_int_t
magma_dznrm2scale( int m, magmaDoubleComplex *r, int lddr, magmaDoubleComplex *drnorm);

#include <cblas.h>

#define PRECISION_z

#define RTOLERANCE     10e-10
#define ATOLERANCE     10e-10

/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    Solves a system of linear equations
       A * X = B
    where A is a complex sparse matrix stored in the GPU memory.
    X and B are complex vectors stored on the GPU memory. 
    This is a GPU implementation of the GMRES method.

    Arguments
    =========

    magma_z_sparse_matrix A                   descriptor for matrix A
    magma_z_vector b                          RHS b vector
    magma_z_vector *x                         solution approximation
    magma_solver_parameters *solver_par       solver parameters

    =====================================================================  */

magma_int_t
magma_zgmres( magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
              magma_solver_parameters *solver_par ){

#define  q(i)     (q.val + (i)*dofs)
#define  H(i,j)  H[(i)   + (j)*(1+ldh)]
#define HH(i,j) HH[(i)   + (j)*ldh]
#define dH(i,j) dH[(i)   + (j)*(1+ldh)]

    //Chronometry
    double tempo1, tempo2;

    // local variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE, c_mone = MAGMA_Z_NEG_ONE;
    magma_int_t dofs = A.num_rows;
    magma_int_t i, j, k, m = 0, iter, restart = min( dofs-1, solver_par->restart );
    magma_int_t ldh = restart+1;
    double rNorm, RNorm = 0.0, den, nom0, r0 = 0.;

    // CPU workspace
    magma_setdevice(0);
    magmaDoubleComplex *H, *HH, *y, *h1;
    magma_zmalloc_pinned( &H, (ldh+1)*ldh );
    magma_zmalloc_pinned( &y, ldh );
    magma_zmalloc_pinned( &HH, ldh*ldh );
    magma_zmalloc_pinned( &h1, ldh );
    // GPU workspace
    magma_z_vector r, q, q_t;
    magma_z_vinit( &r, Magma_DEV, dofs, c_zero );
    magma_z_vinit( &q, Magma_DEV, dofs*(ldh+1), c_zero );
    q_t.memory_location = Magma_DEV; q_t.val = NULL; q_t.num_rows = q_t.nnz = dofs;

    magmaDoubleComplex *dy, *dH = NULL;
    if (MAGMA_SUCCESS != magma_zmalloc( &dy, ldh )) 
        return MAGMA_ERR_DEVICE_ALLOC;
    if (MAGMA_SUCCESS != magma_zmalloc( &dH, (ldh+1)*ldh )) 
        return MAGMA_ERR_DEVICE_ALLOC;
    // GPU stream
    magma_queue_t stream[2];
    magma_event_t event[1];
    magma_queue_create( &stream[0] );
    magma_queue_create( &stream[1] );
    magma_event_create( &event[0] );
    //trace_init( 1, 1, 2, (CUstream_st**)stream );

    //Chronometry 
    magma_device_sync(); tempo1=magma_wtime();
    magmablasSetKernelStream(stream[0]);

    magma_zscal( dofs, c_zero, x->val, 1 );              //  x = 0
    magma_zcopy( dofs, b.val, 1, r.val, 1 );             //  r = b

    r0 = magma_dznrm2( dofs, r.val, 1 );                 //  r0= || r||
    nom0 = r0*r0;
    H(1,0) = MAGMA_Z_MAKE( r0, 0. ); 
    magma_zsetvector(1, &H(1,0), 1, &dH(1,0), 1);

    r0 *= solver_par->epsilon;
    //if (r0 < ATOLERANCE) 
    //    r0 = ATOLERANCE;
    
    //#define ENABLE_TIMER
    #ifdef ENABLE_TIMER
    printf("Iteration: %4d  Norm: %e  Time: %e\n", 0, MAGMA_Z_REAL(H(1,0)*H(1,0)), 0.0);
    double t_spmv1, t_spmv = 0.0;
    double t_orth1, t_orth2, t_gemv_reduce = 0.0, t_orth = 0.0;
    #endif

    for (iter = 0; iter<solver_par->maxiter; iter++) 
        {
            magma_zcopy(dofs, r.val, 1, q(0), 1);                        //  q[0]    = 1.0/||r||
            magma_zscal(dofs, 1./H(1,0), q(0), 1);                     //  (to be fused)

            for(k=1; k<=restart; k++) 
                {

                    #ifdef ENABLE_TIMER
                    magma_device_sync(); t_spmv1=magma_wtime();
                    #endif
                    q_t.val = q(k-1);
                    magmablasSetKernelStream(stream[0]);
                    //trace_gpu_start( 0, 0, "spmv", "spmv" );
                    magma_z_spmv( c_one, A, q_t, c_zero, r );                      //  r       = A q[k] 
                    //trace_gpu_end( 0, 0 );
                    #ifdef ENABLE_TIMER
                    magma_device_sync(); t_spmv += magma_wtime()-t_spmv1;
                    #endif
                    
                    #ifdef ENABLE_TIMER
                    magma_device_sync(); t_orth1=magma_wtime();
                    #endif
                    if (solver_par->gmres_ortho == MAGMA_MGS ) {
                        // modified Gram-Schmidt
                        magmablasSetKernelStream(stream[0]);
                        for (i=1; i<=k; i++) {
                            H(i,k) =magma_zdotc(dofs, q(i-1), 1, r.val, 1);            //  H(i,k) = q[i] . r
                            magma_zaxpy(dofs,-H(i,k), q(i-1), 1, r.val, 1);            //  r      = r - H(i,k) q[i]
                        }
                        H(k+1,k) = MAGMA_Z_MAKE( magma_dznrm2(dofs, r.val, 1), 0. );   //  H(k+1,k) = sqrt(r . r) 
                        if (k < restart) {
                                magma_zcopy(dofs, r.val, 1, q(k), 1);                  //  q[k] = 1.0/H[k][k-1] r
                                magma_zscal(dofs, 1./H(k+1,k), q(k), 1);               //  (to be fused)   
                         }
                    } else if (solver_par->gmres_ortho == MAGMA_FUSED_CGS ) {
                        // fusing zgemv with dznrm2 in classical Gram-Schmidt
                        magmablasSetKernelStream(stream[0]);
                        trace_gpu_start( 0, 0, "copy", "copy" );              // q[k] = r
                        magma_zcopy(dofs, r.val, 1, q(k), 1);  
                        //trace_gpu_end( 0, 0 );

                        //trace_gpu_start( 0, 0, "gemv", "gemv" );
                        #ifdef ENABLE_TIMER
                        magma_device_sync(); t_orth2=magma_wtime();
                        #endif
                                                                              // dH(1:k+1,k) = q[0:k] . r
                        magmablas_zgemv(MagmaTrans, dofs, k+1, c_one, q(0), dofs, r.val, 1, c_zero, &dH(1,k), 1);
                        #ifdef ENABLE_TIMER
                        magma_device_sync(); t_gemv_reduce += magma_wtime()-t_orth2;
                        #endif
                                                                              // r = r - q[0:k-1] dH(1:k,k)
                        magmablas_zgemv(MagmaNoTrans, dofs, k, c_mone, q(0), dofs, &dH(1,k), 1, c_one, r.val, 1);
                        //trace_gpu_end( 0, 0 );

                        //trace_gpu_start( 0, 0, "scal", "scal" );
                                                                              // 1) dH(k+1,k) = sqrt( dH(k+1,k) - dH(1:k,k) )
                        magma_zcopyscale(  dofs, k, r.val, q(k), &dH(1,k) );  // 2) q[k] = q[k] / dH(k+1,k) 
                        //trace_gpu_end( 0, 0 );
                        magma_event_record( event[0], stream[0] );

                        magma_queue_wait_event( stream[1], event[0] );
                        //trace_gpu_start( 0, 1, "get", "get" );
                        magma_zgetvector_async(k+1, &dH(1,k), 1, &H(1,k), 1, stream[1]); // asynch copy dH(1:(k+1),k) to H(1:(k+1),k)
                        //trace_gpu_end( 0, 1 );
                    } else {
                        // classical Gram-Schmidt (default)
                        #ifdef ENABLE_TIMER
                        magma_device_sync(); t_orth2=magma_wtime();
                        #endif
                        // > explicitly calling magmabls
                        magmablasSetKernelStream(stream[0]);
                                                                              // dH(1:k,k) = q[0:k-1] . r
                        magmablas_zgemv(MagmaTrans, dofs, k, c_one, q(0), dofs, r.val, 1, c_zero, &dH(1,k), 1);
                        #ifdef ENABLE_TIMER
                        magma_device_sync(); t_gemv_reduce += magma_wtime()-t_orth2;
                        #endif

                        #ifndef DZNRM2SCALE 
                        // start copying dH(1:k,k) to H(1:k,k)
                        magma_event_record( event[0], stream[0] );
                        magma_queue_wait_event( stream[1], event[0] );
                        magma_zgetvector_async(k, &dH(1,k), 1, &H(1,k), 1, stream[1]);
                        #endif
                                                                              // r = r - q[0:k-1] dH(1:k,k)
                        magmablas_zgemv(MagmaNoTrans, dofs, k, c_mone, q(0), dofs, &dH(1,k), 1, c_one, r.val, 1);

                        #ifdef DZNRM2SCALE
                        magma_zcopy(dofs, r.val, 1, q(k), 1);                 //  q[k] = r / H(k,k-1) 
                        magma_dznrm2scale(dofs, q(k), dofs, &dH(k+1,k) );     //  dH(k+1,k) = sqrt(r . r) and r = r / dH(k+1,k)

                        magma_event_record( event[0], stream[0] );            // start sending dH(1:k,k) to H(1:k,k)
                        magma_queue_wait_event( stream[1], event[0] );        // > can we keep H(k+1,k) on GPU and combine?
                        magma_zgetvector_async(k+1, &dH(1,k), 1, &H(1,k), 1, stream[1]);
                        #else
                        H(k+1,k) = MAGMA_Z_MAKE( magma_dznrm2(dofs, r.val, 1), 0. );   //  H(k+1,k) = sqrt(r . r) 
                        if( k<solver_par->restart ){
                                magmablasSetKernelStream(stream[0]);
                                magma_zcopy(dofs, r.val, 1, q(k), 1);                  //  q[k]    = 1.0/H[k][k-1] r
                                magma_zscal(dofs, 1./H(k+1,k), q(k), 1);               //  (to be fused)   
                         }
                        #endif
                    }
                    #ifdef ENABLE_TIMER
                    magma_device_sync(); t_orth += magma_wtime()-t_orth1;
                    #endif
                }
            magma_queue_sync( stream[1] );
            //trace_cpu_start( 0, "dot", "restart" );
            for(k=1; k<=restart; k++) 
                {
                    /*     Minimization of  || b-Ax ||  in H_k       */ 
                    for (i=1; i<=k; i++) {
                        #if defined(PRECISION_z) || defined(PRECISION_c)
                        cblas_zdotc_sub( i+1, &H(1,k), 1, &H(1,i), 1, &HH(k,i) );
                        #else
                        HH(k,i) = cblas_zdotc(i+1, &H(1,k), 1, &H(1,i), 1);
                        #endif
                    }
                    
                    h1[k] = H(1,k)*H(1,0);
                    
                    if (k != 1)
                        for (i=1; i<k; i++) {
                            for (m=i+1; m<k; m++)
                                HH(k,m) -= HH(k,i) * HH(m,i);

                            HH(k,k) -= HH(k,i) * HH(k,i) / HH(i,i);
                            HH(k,i) = HH(k,i)/HH(i,i);
                            h1[k] -= h1[i] * HH(k,i);   
                        }    

                    y[k] = h1[k]/HH(k,k); 
                    if (k != 1)  
                        for (i=k-1; i>=1; i--) {
                            y[i] = h1[i]/HH(i,i);
                            for (j=i+1; j<=k; j++)
                                y[i] -= y[j] * HH(j,i);
                        }
                    
                    m = k;
                    
                    rNorm = fabs(MAGMA_Z_REAL(H(k+1,k)));
                }
            //trace_cpu_end( 0 );
            
            /*   Update the current approximation: x += Q y  */
            //trace_gpu_start( 0, 0, "set", "set" );
            magma_zsetmatrix_async(m, 1, y+1, m, dy, m, stream[0]);
            //trace_gpu_end( 0, 0 );
            magmablasSetKernelStream(stream[0]);
            //trace_gpu_start( 0, 0, "gemv", "gemv" );
            magma_zgemv(MagmaNoTrans, dofs, m, c_one, q(0), dofs, dy, 1, c_one, x->val, 1); 
            //trace_gpu_end( 0, 0 );

            #ifdef ENABLE_TIMER
            magma_device_sync(); t_spmv1=magma_wtime();
            #endif
            //trace_gpu_start( 0, 0, "spmv", "spmv" );
            magma_z_spmv( c_mone, A, *x, c_zero, r );                  //  r = - A * x
            //trace_gpu_end( 0, 0 );
            #ifdef ENABLE_TIMER
            magma_device_sync(); t_spmv += magma_wtime() - t_spmv1;
            #endif

            magma_zaxpy(dofs, c_one, b.val, 1, r.val, 1);              //  r = r + b
            H(1,0) = MAGMA_Z_MAKE( magma_dznrm2(dofs, r.val, 1), 0. ); //  RNorm = H[1][0] = || r ||
            RNorm = MAGMA_Z_REAL( H(1,0) );
            
            #ifdef ENABLE_TIMER
            //Chronometry  
            magma_device_sync(); tempo2=magma_wtime();
            printf("Iteration: %4d  Norm: %e  Time: %.2lf, GEMV_reduce: %.2lf (%.2lf%%) Orth: %.2lf (%.2lf%%) SpMV: %.2lf (%.2lf%%)\n", 
                    iter+1, RNorm*RNorm, tempo2-tempo1, 
                    t_gemv_reduce, 100.0*t_gemv_reduce/(tempo2-tempo1), 
                    t_orth, 100.0*t_orth/(tempo2-tempo1), 
                    t_spmv, 100.0*t_spmv/(tempo2-tempo1));
            #endif
            
            if (fabs(RNorm*RNorm) < r0) break;    
        }
    
  
    printf( "\n" );
    printf( "      (r_0, r_0) = %e\n", nom0 );
    printf( "      (r_N, r_N) = %e\n", RNorm*RNorm);
    printf( "      Number of GMRES restarts: %d\n", iter);
    #ifndef ENABLE_TIMER
    magma_device_sync(); tempo2=magma_wtime();
    printf( "      Time: %.2lf\n",tempo2-tempo1 );
    #endif
    if (solver_par->epsilon == RTOLERANCE) {
        magma_z_spmv( c_one, A, *x, c_zero, r );                       // q_t = A x
        magma_zaxpy(dofs,  c_mone, b.val, 1, r.val, 1);                // r = r - b
        den = magma_dznrm2(dofs, r.val, 1);                            // den = || r ||
        printf( "      || r_N ||   = %f\n", den);
        solver_par->residual = (double)(den);
    }
    printf( "\n" );
    solver_par->numiter = iter;
    //trace_finalize( "zgmres.svg","trace.css" );

    // free GPU streams and events
    magma_queue_destroy( stream[0] );
    magma_queue_destroy( stream[1] );
    magma_event_destroy( event[0] );
    magmablasSetKernelStream(NULL);
    // free pinned memory
    magma_free_pinned( H );
    magma_free_pinned( y );
    magma_free_pinned( HH );
    magma_free_pinned( h1 );
    // free GPU memory
    magma_free(dy); 
    if (dH != NULL ) magma_free(dH); 
    magma_z_vfree(&r);
    magma_z_vfree(&q);

    return MAGMA_SUCCESS;
}   /* magma_zgmres */

