#include <stdio.h>
#include <math.h>
#include "magmablas.h"
#include "magma.h"
#include "cublas.h"
#include "cuda.h"

void magma_dpotrs_gpu( char *UPLO , int N , int NRHS, double *A , int LDA ,double *B, int LDB, int *INFO){
                if( *UPLO =='U'){
                         magmablas_dtrsm('L','U','T','N', N , NRHS,   A , LDA , B , LDB );
                         magmablas_dtrsm('L','U','N','N', N , NRHS,   A , LDA , B , LDB );
                }
                else{
                         magmablas_dtrsm('L','L','N','N', N , NRHS,  A , LDA , B , LDB );
                         magmablas_dtrsm('L','L','T','N', N , NRHS,  A , LDA , B , LDB );
                }
}
