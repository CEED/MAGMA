#include <stdio.h>

#include "magma_v2.h"
#include "testings.h"

int main( int argc, char** argv )
{
    magma_init();
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        printf( "m %5ld, n %5ld, k %5ld\n",
                long(opts.msize[itest]), long(opts.nsize[itest]), long(opts.ksize[itest]) );
    }
    printf( "\n" );
    
    printf( "ntest    %ld\n", long(opts.ntest) );
    printf( "\n" );
    
    printf( "nb       %ld\n", long(opts.nb)       );
    printf( "nrhs     %ld\n", long(opts.nrhs)     );
    printf( "nqueue   %ld\n", long(opts.nqueue)   );
    printf( "ngpu     %ld\n", long(opts.ngpu)     );
    printf( "niter    %ld\n", long(opts.niter)    );
    printf( "nthread  %ld\n", long(opts.nthread)  );
    printf( "itype    %ld\n", long(opts.itype)    );
    printf( "verbose  %ld\n", long(opts.verbose)  );
    printf( "\n" );
    
    printf( "check    %s\n", (opts.check  ? "true" : "false") );
    printf( "lapack   %s\n", (opts.lapack ? "true" : "false") );
    printf( "warmup   %s\n", (opts.warmup ? "true" : "false") );
    printf( "\n" );
    
    printf( "uplo     %3d (%s)\n", opts.uplo,   lapack_uplo_const(  opts.uplo   ));
    printf( "transA   %3d (%s)\n", opts.transA, lapack_trans_const( opts.transA ));
    printf( "transB   %3d (%s)\n", opts.transB, lapack_trans_const( opts.transB ));
    printf( "side     %3d (%s)\n", opts.side,   lapack_side_const(  opts.side   ));
    printf( "diag     %3d (%s)\n", opts.diag,   lapack_diag_const(  opts.diag   ));
    printf( "jobz     %3d (%s)\n", opts.jobz,   lapack_vec_const(   opts.jobz   ));
    printf( "jobvr    %3d (%s)\n", opts.jobvr,  lapack_vec_const(   opts.jobvr  ));
    printf( "jobvl    %3d (%s)\n", opts.jobvl,  lapack_vec_const(   opts.jobvl  ));
    
    for( auto iter = opts.svd_work.begin(); iter < opts.svd_work.end(); ++iter ) {
        printf( "svd_work %ld\n", long(*iter) );
    }
    for( auto iter = opts.jobu.begin(); iter < opts.jobu.end(); ++iter ) {
        printf( "jobu     %3d (%s)\n", *iter, lapack_vec_const( *iter ));
    }
    for( auto iter = opts.jobv.begin(); iter < opts.jobv.end(); ++iter ) {
        printf( "jobvt    %3d (%s)\n", *iter, lapack_vec_const( *iter ));
    }
    
    opts.cleanup();
    magma_finalize();
    
    return 0;
}
