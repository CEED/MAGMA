/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar
       @precisions normal z -> c d s

*/
#include "common_magma.h"
#define COMPLEX


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t
magma_zbulge_get_lq2(magma_int_t n, magma_int_t threads, magma_int_t wantz)
{
    if (wantz == 0)
        return 2*n*2;

    magma_int_t nb = magma_zbulge_get_nb(n, threads);
    magma_int_t Vblksiz = magma_zbulge_get_Vblksiz(n, nb, threads);
    magma_int_t ldv = nb + Vblksiz;
    magma_int_t ldt = Vblksiz;

    return magma_bulge_get_blkcnt(n, nb, Vblksiz) * Vblksiz * (ldt + ldv + 1);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" void
magma_zbulge_get_VTsiz(magma_int_t n, magma_int_t nb, magma_int_t threads,
        magma_int_t *Vblksiz, magma_int_t *ldv, magma_int_t *ldt)
{
    Vblksiz[0] = magma_zbulge_get_Vblksiz(n, nb, threads);
    ldv[0]     = nb + Vblksiz[0];
    ldt[0]     = Vblksiz[0];
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t
magma_zbulge_getstg2size(magma_int_t n, magma_int_t nb, magma_int_t wantz,
                         magma_int_t Vblksiz, magma_int_t ldv, magma_int_t ldt,
                         magma_int_t *blkcnt, magma_int_t *sizTAU2,
                         magma_int_t *sizT2, magma_int_t *sizV2)
{
    blkcnt[0]  = magma_bulge_get_blkcnt(n, nb, Vblksiz);
    sizTAU2[0] = wantz == 0 ? 2*n :  blkcnt[0]*Vblksiz;
    sizV2[0]   = wantz == 0 ? 2*n :  blkcnt[0]*Vblksiz*ldv;
    sizT2[0]   = wantz == 0 ? 0   :  blkcnt[0]*Vblksiz*ldt;
    return sizTAU2[0] + sizT2[0] + sizV2[0];
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t
magma_zbulge_getlwstg2(magma_int_t n, magma_int_t threads, magma_int_t wantz,
                       magma_int_t *Vblksiz, magma_int_t *ldv, magma_int_t *ldt,
                       magma_int_t *blkcnt, magma_int_t *sizTAU2,
                       magma_int_t *sizT2, magma_int_t *sizV2)
{
    magma_int_t nb      = magma_zbulge_get_nb(n, threads);
    magma_zbulge_get_VTsiz(n, nb, threads, Vblksiz, ldv, ldt);
    return magma_zbulge_getstg2size(n, nb, wantz, Vblksiz[0], ldv[0], ldt[0], blkcnt, sizTAU2, sizT2, sizV2);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" void
magma_zheevdx_getworksize(magma_int_t n, magma_int_t threads,
        magma_int_t wantz,
        magma_int_t *lwmin,
        #ifdef COMPLEX
        magma_int_t *lrwmin,
        #endif
        magma_int_t *liwmin)
{
    magma_int_t lda2=0;
    magma_int_t Vblksiz;
    magma_int_t ldv;
    magma_int_t ldt;
    magma_int_t blkcnt;
    magma_int_t sizTAU2;
    magma_int_t sizT2;
    magma_int_t sizV2;
    magma_int_t nb     = magma_zbulge_get_nb( n, threads );
    magma_int_t lwstg1 = magma_bulge_getlwstg1( n, nb, &lda2 );
    magma_int_t lwstg2 = magma_zbulge_getlwstg2( n, threads, wantz, &Vblksiz, &ldv, &ldt, &blkcnt, &sizTAU2, &sizT2, &sizV2 );

    #ifdef COMPLEX
    if (wantz) {
        *lwmin  = lwstg2 + 2*n + max(lwstg1, n*n);
        *lrwmin = 1 + 5*n + 2*n*n;
        *liwmin = 5*n + 3;
    } else {
        *lwmin  = lwstg2 + n + lwstg1;
        *lrwmin = n;
        *liwmin = 1;
    }
    #else
    if (wantz) {
        *lwmin  = lwstg2 + 1 + 6*n + max(lwstg1, 2*n*n);
        *liwmin = 5*n + 3;
    } else {
        *lwmin  = lwstg2 + 2*n + lwstg1;
        *liwmin = 1;
    }
    #endif
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
