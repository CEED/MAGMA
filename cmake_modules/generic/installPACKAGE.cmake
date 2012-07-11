###
#
# @file          : installPACKAGE.cmake
#
# @description   :
#
# @version       :
# @created by    : Cedric Castagnede
# @creation date : 09-03-2012
# @last modified : mar. 12 juin 2012 16:43:45 CEST
#
###

# Macro to call installation
# --------------------------
MACRO(INSTALL_PACKAGE _NAME _MODE)

    STRING(TOUPPER "${_NAME}" _NAMEVAR)

    IF("BLAS" MATCHES "${_NAMEVAR}")
        INCLUDE(installBLAS)
        INSTALL_BLAS("${_MODE}")

    ELSEIF("LAPACK" MATCHES "${_NAMEVAR}")
        INCLUDE(installLAPACK)
        INSTALL_LAPACK("${_MODE}")

    ELSEIF("CBLAS" MATCHES "${_NAMEVAR}")
        INCLUDE(installCBLAS)
        INSTALL_CBLAS("${_MODE}")

    ELSEIF("LAPACKE" MATCHES "${_NAMEVAR}")
        INCLUDE(installLAPACKE)
        INSTALL_LAPACKE("${_MODE}")

    ELSEIF("PLASMA" MATCHES "${_NAMEVAR}")
        INCLUDE(installPLASMA)
        INSTALL_PLASMA("${_MODE}")

    ELSEIF("MPI" MATCHES "${_NAMEVAR}")
        INCLUDE(installOPENMPI)
        INSTALL_OPENMPI("${_MODE}")

    ELSEIF("HWLOC" MATCHES "${_NAMEVAR}")
        INCLUDE(installHWLOC)
        INSTALL_HWLOC("${_MODE}")

    ELSEIF("FXT" MATCHES "${_NAMEVAR}")
        INCLUDE(installFXT)
        INSTALL_FXT("${_MODE}")

    ELSEIF("STARPU" MATCHES "${_NAMEVAR}")
        INCLUDE(installSTARPU)
        INSTALL_STARPU("${_MODE}")

    ELSE()
        MESSAGE(FATAL_ERROR "DEV_ERROR - Macro INSTALL_PACAKGE")

    ENDIF()

ENDMACRO(INSTALL_PACKAGE)

###
### END installPACKAGE.cmake
###
