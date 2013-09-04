/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> c d s
       @author Hartwig Anzt

*/
#include "common_magma.h"

#define BLOCK_SIZE 512

#define PRECISION_z


/* ---------------------------------------------------------------------------------- */

__global__ void 
magma_zbicgmerge1_kernel(  
                    int n, 
                    magmaDoubleComplex *skp,
                    magmaDoubleComplex *v, 
                    magmaDoubleComplex *r, 
                    magmaDoubleComplex *p ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    magmaDoubleComplex beta=skp[1];
    magmaDoubleComplex omega=skp[2];
    if( i<n ){
        p[i] =  r[i] + beta * ( p[i] - omega * v[i] );

    }

}

/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    Mergels multiple operations into one kernel:

    magma_zscal( dofs, beta, p.val, 1 );                                    //p = beta*p
    magma_zaxpy( dofs, c_mone * omega * beta, v.val, 1 , p.val, 1 );        //p = p-omega*beta*v
    magma_zaxpy( dofs, c_one, r.val, 1, p.val, 1 );                         //p = p+r
    
    -> p = r + beta * ( p - omega * v ) 

    Arguments
    =========

    int n                               dimension n
    magmaDoubleComplex beta             scalar 
    magmaDoubleComplex omega            scalar
    magmaDoubleComplex *v               input v
    magmaDoubleComplex *r               input r
    magmaDoubleComplex *p               input/output p

    =====================================================================  */

extern "C" int
magma_zbicgmerge1(  int n, 
                    magmaDoubleComplex *skp,
                    magmaDoubleComplex *v, 
                    magmaDoubleComplex *r, 
                    magmaDoubleComplex *p ){

    
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( (n+BLOCK_SIZE-1)/BLOCK_SIZE );
    magma_zbicgmerge1_kernel<<<Gs, Bs, 0>>>( n, skp, v, r, p );

   return MAGMA_SUCCESS;
}

/* ---------------------------------------------------------------------------------- */

__global__ void 
magma_zbicgmerge2_kernel(  
                    int n, 
                    magmaDoubleComplex *skp, 
                    magmaDoubleComplex *r,
                    magmaDoubleComplex *v, 
                    magmaDoubleComplex *s ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    magmaDoubleComplex alpha=skp[0];
    if( i<n ){
        s[i] =  r[i] - alpha * v[i] ;
    }

}

/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    Mergels multiple operations into one kernel:

    magma_zcopy( dofs, r.val, 1 , s.val, 1 );                                // s=r
    magma_zaxpy( dofs, c_mone * alpha, v.val, 1 , s.val, 1 );                // s=s-alpha*v
        
    -> s = r - alpha * v

    Arguments
    =========

    int n                               dimension n
    magmaDoubleComplex alpha            scalar 
    magmaDoubleComplex *r               input r
    magmaDoubleComplex *v               input v
    magmaDoubleComplex *s               input/output s

    =====================================================================  */

extern "C" int
magma_zbicgmerge2(  int n, 
                    magmaDoubleComplex *skp, 
                    magmaDoubleComplex *r,
                    magmaDoubleComplex *v, 
                    magmaDoubleComplex *s ){

    
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( (n+BLOCK_SIZE-1)/BLOCK_SIZE );

    magma_zbicgmerge2_kernel<<<Gs, Bs, 0>>>( n, skp, r, v, s );

   return MAGMA_SUCCESS;
}

/* ---------------------------------------------------------------------------------- */

__global__ void 
magma_zbicgmerge3_kernel(  
                    int n, 
                    magmaDoubleComplex *skp, 
                    magmaDoubleComplex *p,
                    magmaDoubleComplex *se,
                    magmaDoubleComplex *t,
                    magmaDoubleComplex *x, 
                    magmaDoubleComplex *r ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    magmaDoubleComplex alpha=skp[0];
    magmaDoubleComplex omega=skp[2];
    if( i<n ){
        magmaDoubleComplex s;
        s = se[i];
        x[i] = x[i] + alpha * p[i] + omega * s;
        r[i] = s - omega * t[i];
    }

}

/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    Mergels multiple operations into one kernel:

        magma_zaxpy( dofs, alpha, p.val, 1 , x->val, 1 );                        // x=x+alpha*p
        magma_zaxpy( dofs, omega, s.val, 1 , x->val, 1 );                        // x=x+omega*s
        magma_zcopy( dofs, s.val, 1 , r.val, 1 );                                // r=s
        magma_zaxpy( dofs, c_mone * omega, t.val, 1 , r.val, 1 );                // r=r-omega*t
        
    -> x = x + alpha * p + omega * s
    -> r = s - omega * t

    Arguments
    =========

    int n                               dimension n
    magmaDoubleComplex alpha            scalar 
    magmaDoubleComplex omega            scalar 
    magmaDoubleComplex *p               input p
    magmaDoubleComplex *s               input s
    magmaDoubleComplex *t               input t
    magmaDoubleComplex *x               input/output x
    magmaDoubleComplex *r               input/output r

    =====================================================================  */

extern "C" int
magma_zbicgmerge3(  int n, 
                    magmaDoubleComplex *skp,
                    magmaDoubleComplex *p,
                    magmaDoubleComplex *s,
                    magmaDoubleComplex *t,
                    magmaDoubleComplex *x, 
                    magmaDoubleComplex *r ){

    
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( (n+BLOCK_SIZE-1)/BLOCK_SIZE );
    magma_zbicgmerge3_kernel<<<Gs, Bs, 0>>>( n, skp, p, s, t, x, r );

   return MAGMA_SUCCESS;
}

/* ---------------------------------------------------------------------------------- */

__global__ void 
magma_zbicgmerge4_kernel_1(  
                    magmaDoubleComplex *skp ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if( i==0 ){
        magmaDoubleComplex tmp = skp[0];
        skp[0] = skp[4]/tmp;
    }
}

__global__ void 
magma_zbicgmerge4_kernel_2(  
                    magmaDoubleComplex *skp ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if( i==0 ){
        skp[2] = skp[6]/skp[7];
        skp[3] = skp[4];
    }
}

__global__ void 
magma_zbicgmerge4_kernel_3(  
                    magmaDoubleComplex *skp ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if( i==0 ){
        magmaDoubleComplex tmp1 = skp[4]/skp[3];
        magmaDoubleComplex tmp2 = skp[0] / skp[2];
        skp[1] =  tmp1*tmp2;
        //skp[1] =  skp[4]/skp[3] * skp[0] / skp[2];

    }
}

/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    Performs some parameter operations for the BiCGSTAB with scalars on GPU.

    Arguments
    =========

    int type                            kernel type
    magmaDoubleComplex *skp             vector with parameters

    =====================================================================  */

extern "C" int
magma_zbicgmerge4(  int type, 
                    magmaDoubleComplex *skp ){

    dim3 Bs( 2 );
    dim3 Gs( 1 );
    if( type == 1 )
        magma_zbicgmerge4_kernel_1<<<Gs, Bs, 0>>>( skp );
    else if( type == 2 )
        magma_zbicgmerge4_kernel_2<<<Gs, Bs, 0>>>( skp );
    else if( type == 3 )
        magma_zbicgmerge4_kernel_3<<<Gs, Bs, 0>>>( skp );
    else
        printf("error: no kernel called\n");

   return MAGMA_SUCCESS;
}

