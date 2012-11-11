###
#
#  @file installBLAS.cmake
#
#  @project MORSE
#  MORSE is a software package provided by:
#     Inria Bordeaux - Sud-Ouest,
#     Univ. of Tennessee,
#     Univ. of California Berkeley,
#     Univ. of Colorado Denver.
#
#  @version 0.1.0
#  @author Cedric Castagnede
#  @date 13-07-2012
#
###

MACRO(INSTALL_BLAS _MODE)

    # Get the implementation of BLAS
    # ------------------------------
    INCLUDE(infoBLAS)
    BLAS_DEFINE_DEFAULT()

    # Start the installation of BLAS
    # ------------------------------
    IF("${MORSE_BLAS_DEFAULT_VALUE} "MATCHES "EIGEN")
        INCLUDE(installEIGEN)
        INSTALL_EIGEN("${_MODE}")

    ELSEIF("${MORSE_BLAS_DEFAULT_VALUE}" MATCHES "REFBLAS")
        INCLUDE(installREFBLAS)
        INSTALL_REFBLAS("${_MODE}")

    ELSE()
        MESSAGE(FATAL_ERROR "DEV_ERROR - Macro INSTALL_BLAS")

    ENDIF()

ENDMACRO(INSTALL_BLAS)

##
## @end file installBLAS.cmake
##
