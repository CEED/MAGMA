/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Mark Gates
*/

#include <complex>

#include "magma_operators.h"

// ----------------------------------------
magmaDoubleComplex magma_zsqrt( magmaDoubleComplex a )
{
    std::complex<double> b = std::sqrt( std::complex<double>( real(a), imag(a) ));
    return MAGMA_Z_MAKE( real(b), imag(b) );
}


// ----------------------------------------
magmaFloatComplex magma_csqrt( magmaFloatComplex a )
{
    std::complex<float> b = std::sqrt( std::complex<float>( real(a), imag(a) ));
    return MAGMA_C_MAKE( real(b), imag(b) );
}
