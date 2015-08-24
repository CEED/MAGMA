#include "common_magma.h"

// ========================================
// complex support

// --------------------
// propagates inf and nan correctly:
//     abs( [xxx, nan] ) == nan, for any xxx
//     abs( [nan, xxx] ) == nan, for any xxx
//     abs( [inf, xxx] ) == inf, for any xxx except nan
//     abs( [xxx, inf] ) == inf, for any xxx except nan
extern "C" double
magma_cabs(magmaDoubleComplex z)
{
    double x = fabs( real( z ));
    double y = fabs( imag( z ));
    double big, little;  // "small" reserved in Windows. Ugh.
    if ( x > y ) {
        big    = x;
        little = y;
    }
    else {
        big    = y;
        little = x;
    }
    if ( big == 0 || isinf(big) ) {
        return big + little;  // add to propagate nan
    }
    little /= big;
    return big * sqrt( 1 + little*little );
}

// --------------------
extern "C" float
magma_cabsf(magmaFloatComplex z)
{
    float x = fabs( real( z ));
    float y = fabs( imag( z ));
    float big, little;
    if ( x > y ) {
        big    = x;
        little = y;
    }
    else {
        big    = y;
        little = x;
    }
    if ( big == 0 || isinf(big) ) {
        return big + little;  // add to propagate nan
    }
    little /= big;
    return big * sqrt( 1 + little*little );
}
