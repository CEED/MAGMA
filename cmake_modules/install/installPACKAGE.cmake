###
#
#  @file installPACKAGE.cmake
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

INCLUDE(BuildSystemTools)
INCLUDE(infoPACKAGE)
INCLUDE(FindDependencies)
INCLUDE(populatePACKAGE)
INCLUDE(downloadPACKAGE)

# Macro to call installation
# --------------------------
MACRO(INSTALL_PACKAGE _NAME _MODE)

    # Define input variable to uppercase
    # ----------------------------------
    STRING(TOUPPER "${_NAME}" _UPIPVAR)
    STRING(TOLOWER "${_NAME}" _LOWIPVAR)

    IF(NOT DEFINED ${_MODE})
        # Message about install
        # ---------------------
        #MESSAGE(STATUS "Installing ${_UPIPVAR} - ${_MODE}")

        # Looking for the way to get package
        # ----------------------------------
        INFO_INSTALL_PACKAGE(${_UPIPVAR})
        DEFINE_DOWNLOAD_PACKAGE("${_LOWIPVAR}" "${_MODE}")

        # Looking for dependencies
        # ------------------------
        # Put _UPFDVAR in GIP_MEMORY for recursivity
        LIST(APPEND GIP_MEMORY_1 ${_UPIPVAR})
        LIST(APPEND GIP_MEMORY_2 ${_LOWIPVAR})

        # Find dependencies
        FIND_DEPENDENCIES(${_UPIPVAR})

        # Get back _UPFDVAR from GIP_MEMORY_1
        LIST(LENGTH GIP_MEMORY_1 GIP_SIZE)
        MATH(EXPR IEP_ID "${GIP_SIZE}-1")
        LIST(GET GIP_MEMORY_1 ${IEP_ID} _UPIPVAR)
        LIST(REMOVE_AT GIP_MEMORY_1 ${IEP_ID})

        # Get back _UPFDVAR from GIP_MEMORY_2
        LIST(LENGTH GIP_MEMORY_2 GIP_SIZE)
        MATH(EXPR IEP_ID "${GIP_SIZE}-1")
        LIST(GET GIP_MEMORY_2 ${IEP_ID} _LOWIPVAR)
        LIST(REMOVE_AT GIP_MEMORY_2 ${IEP_ID})

        # Looking for the script to install _UPIPVAR
        # ----------------------------------------
        IF("^BLAS$" STREQUAL "^${_UPIPVAR}$")
            INCLUDE(installBLAS)
            INSTALL_BLAS("${_MODE}")

        ELSEIF("^TMG$" STREQUAL "^${_UPIPVAR}$")
            INCLUDE(installTMG)
            INSTALL_TMG("${_MODE}")

        ELSEIF("^LAPACK$" STREQUAL "^${_UPIPVAR}$")
            INCLUDE(installLAPACK)
            INSTALL_LAPACK("${_MODE}")

        ELSEIF("^CBLAS$" STREQUAL "^${_UPIPVAR}$")
            INCLUDE(installCBLAS)
            INSTALL_CBLAS("${_MODE}")

        ELSEIF("^LAPACKE$" STREQUAL "^${_UPIPVAR}$")
            INCLUDE(installLAPACKE)
            INSTALL_LAPACKE("${_MODE}")

        ELSEIF("^PLASMA$" STREQUAL "^${_UPIPVAR}$")
            INCLUDE(installPLASMA)
            INSTALL_PLASMA("${_MODE}")

        ELSEIF("^MPI$" STREQUAL "^${_UPIPVAR}$")
            INCLUDE(installOPENMPI)
            INSTALL_OPENMPI("${_MODE}")

        ELSEIF("^HWLOC$" STREQUAL "^${_UPIPVAR}$")
            INCLUDE(installHWLOC)
            INSTALL_HWLOC("${_MODE}")

        ELSEIF("^FXT$" STREQUAL "^${_UPIPVAR}$")
            INCLUDE(installFXT)
            INSTALL_FXT("${_MODE}")

        ELSEIF("^STARPU$" STREQUAL "^${_UPIPVAR}$")
            INCLUDE(installSTARPU)
            INSTALL_STARPU("${_MODE}")

        ELSEIF("^QUARK$" STREQUAL "^${_UPIPVAR}$")
            INCLUDE(installQUARK)
            INSTALL_QUARK("${_MODE}")

        ELSE()
            MESSAGE(FATAL_ERROR "DEV_ERROR - Macro INSTALL_PACKAGE - <${_UPIPVAR}|${_MODE}>")

        ENDIF()

        # Populate the CMake Build System
        # -------------------------------
        POPULATE_COMPILE_SYSTEM(${_UPIPVAR})

    ELSE()
        MESSAGE(STATUS "Installing ${_UPIPVAR} - already done")

    ENDIF()

ENDMACRO(INSTALL_PACKAGE)

##
## @end file installPACKAGE.cmake
##
