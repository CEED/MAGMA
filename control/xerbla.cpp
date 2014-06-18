#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "common_magma.h"

/**
    Purpose
    -------
    magma_xerbla is an error handler for the MAGMA routines.
    It is called by a MAGMA routine if an input parameter has an
    invalid value. It prints an error message.

    Installers may consider modifying it to
    call system-specific exception-handling facilities.

    Arguments
    ---------
    @param[in]
    srname  CHAR*
            The name of the routine which called XERBLA.
            In C/C++ it is convenient to use __func__.

    @param[in]
    info    INTEGER
            The position of the invalid parameter in the parameter list
            of the calling routine.

    @ingroup magma_aux
    ********************************************************************/
extern "C"
void magma_xerbla(const char *srname , magma_int_t info)
{
    fprintf( stderr, "On entry to %s, parameter %d had an illegal value\n",
             srname, info );
}
