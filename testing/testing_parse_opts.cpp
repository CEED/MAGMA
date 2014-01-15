#include "testings.h"

int main( int argc, char** argv )
{
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    for( int i = 0; i < opts.ntest; ++i ) {
        printf( "m %5d, n %5d, k %5d\n",
                (int) opts.msize[i], (int) opts.nsize[i], (int) opts.ksize[i] );
    }
    printf( "\n" );
    
    printf( "ntest    %d\n", (int) opts.ntest );
    printf( "mmax     %d\n", (int) opts.mmax  );
    printf( "nmax     %d\n", (int) opts.nmax  );
    printf( "kmax     %d\n", (int) opts.kmax  );
    printf( "\n" );
    
    printf( "nb       %d\n", (int) opts.nb       ); 
    printf( "nrhs     %d\n", (int) opts.nrhs     );
    printf( "nstream  %d\n", (int) opts.nstream  );
    printf( "ngpu     %d\n", (int) opts.ngpu     );
    printf( "niter    %d\n", (int) opts.niter    );
    printf( "nthread  %d\n", (int) opts.nthread  );
    printf( "itype    %d\n", (int) opts.itype    );
    printf( "svd_work %d\n", (int) opts.svd_work );
    printf( "\n" );
    
    printf( "check    %s\n", (opts.check  ? "true" : "false") );
    printf( "lapack   %s\n", (opts.lapack ? "true" : "false") );
    printf( "warmup   %s\n", (opts.warmup ? "true" : "false") );
    printf( "all      %s\n", (opts.all    ? "true" : "false") );
    printf( "\n" );
    
    printf( "uplo     %c (%s)\n", opts.uplo  , lapack_const( opts.uplo   ));
    printf( "transA   %c (%s)\n", opts.transA, lapack_const( opts.transA ));
    printf( "transB   %c (%s)\n", opts.transB, lapack_const( opts.transB ));
    printf( "side     %c (%s)\n", opts.side  , lapack_const( opts.side   ));
    printf( "diag     %c (%s)\n", opts.diag  , lapack_const( opts.diag   ));
    printf( "jobu     %c (%s)\n", opts.jobu  , lapack_const( opts.jobu   ));
    printf( "jobvt    %c (%s)\n", opts.jobvt , lapack_const( opts.jobvt  ));
    printf( "jobz     %c (%s)\n", opts.jobz  , lapack_const( opts.jobz   ));
    printf( "jobvr    %c (%s)\n", opts.jobvr , lapack_const( opts.jobvr  ));
    printf( "jobvl    %c (%s)\n", opts.jobvl , lapack_const( opts.jobvl  ));
    
    return 0;
}
