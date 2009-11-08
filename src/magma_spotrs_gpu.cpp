#include <stdio.h>
#include <math.h>
#include "magmablas.h"
#include "magma.h"
#include "cublas.h"
#include "cuda.h"

void magma_spotrs_gpu( char *UPLO , int N , int NRHS, float *A , int LDA ,float *B, int LDB, int *INFO){
                if( *UPLO =='U'){
                         magmablas_strsm('L','U','T','N', N , NRHS,   A , LDA , B , LDB );
                         magmablas_strsm('L','U','N','N', N , NRHS,   A , LDA , B , LDB );
                }
                else{
                         magmablas_strsm('L','L','N','N', N , NRHS,  A , LDA , B , LDB );
                         magmablas_strsm('L','L','T','N', N , NRHS,  A , LDA , B , LDB );
                }
}
