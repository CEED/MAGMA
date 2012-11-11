###
#
#  @file installTMG.cmake
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

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
INCLUDE(installExternalPACKAGE)

MACRO(INSTALL_TMG _MODE)

    # Define prefix paths
    # -------------------
    IF(MORSE_SEPARATE_PROJECTS)
        SET(TMG_PATH ${CMAKE_INSTALL_PREFIX}/tmg)
    ELSE(MORSE_SEPARATE_PROJECTS)
        SET(TMG_PATH ${CMAKE_INSTALL_PREFIX})
    ENDIF(MORSE_SEPARATE_PROJECTS)

    # Create make.inc
    # ---------------
    IF(EXISTS ${CMAKE_BINARY_DIR}/tmg_make.inc)
        FILE(REMOVE ${CMAKE_BINARY_DIR}/tmg_make.inc)
    ENDIF()
    STRING(TOUPPER "${CMAKE_BUILD_TYPE}" TYPE)
    FILE(APPEND ${CMAKE_BINARY_DIR}/tmg_make.inc
"
SHELL      = /bin/sh

FORTRAN    = ${CMAKE_Fortran_COMPILER}

OPTS       = ${CMAKE_Fortran_FLAGS_${TYPE}}
OPTS      += ${CMAKE_Fortran_FLAGS}
OPTS      += ${CMAKE_Fortran_LDFLAGS}

DRVOPTS    = ${CMAKE_Fortran_FLAGS_${TYPE}}
DRVOPTS   += ${CMAKE_Fortran_FLAGS}
DRVOPTS   += ${CMAKE_Fortran_LDFLAGS}

NOOPT      = ${CMAKE_Fortran_FLAGS_${TYPE}}
NOOPT     += ${CMAKE_Fortran_FLAGS}

LOADER     = ${CMAKE_Fortran_COMPILER}
LOADOPTS   = ${CMAKE_Fortran_FLAGS_${TYPE}}
LOADOPTS  += ${CMAKE_Fortran_LDFLAGS}

TIMER      = INT_CPU_TIME

ARCH       = ${CMAKE_AR}
ARCHFLAGS  = cr
RANLIB     = ${CMAKE_RANLIB}

BLASLIB    = ${BLAS_LDFLAGS}
XBLASLIB   = 
LAPACKLIB  = liblapack.${MORSE_LIBRARY_EXTENSION}
TMGLIB     = libtmg.${MORSE_LIBRARY_EXTENSION}
EIGSRCLIB  = libeigsrc.${MORSE_LIBRARY_EXTENSION}
LINSRCLIB  = liblinsrc.${MORSE_LIBRARY_EXTENSION}
")
    IF(BUILD_SHARED_LIBS)
        FILE(APPEND ${CMAKE_BINARY_DIR}/tmg_make.inc "
OPTS      += -fPIC
NOOPT     += -fPIC
DRVOPTS   += -fPIC
ARCHFLAGS  = rcs
")
    ENDIF(BUILD_SHARED_LIBS)

    # Define steps of installation
    # ----------------------------
    SET(TMG_SOURCE_PATH ${CMAKE_BINARY_DIR}/externals/tmg)
    SET(TMG_BUILD_PATH  ${CMAKE_BINARY_DIR}/externals/tmg)
    SET(TMG_SOURCE_PATH ${CMAKE_BINARY_DIR}/externals/tmg)
    SET(TMG_CONFIG_CMD  ${CMAKE_COMMAND} -E copy
                             ${CMAKE_BINARY_DIR}/tmg_make.inc
                             ${TMG_BUILD_PATH}/make.inc)
    SET(TMG_BUILD_CMD     ${CMAKE_MAKE_PROGRAM} tmglib)
    SET(TMG_INSTALL_CMD ${CMAKE_COMMAND} -E make_directory ${TMG_PATH})

    # Define additional step
    # ----------------------
    UNSET(TMG_ADD_INSTALL_STEP)
    FOREACH(_task lib)
        LIST(APPEND TMG_ADD_INSTALL_STEP tmg_create_${_task}_path)
        SET(tmg_create_${_task}_path_CMD ${CMAKE_COMMAND} -E make_directory ${TMG_PATH}/${_task})
        SET(tmg_create_${_task}_path_DIR ${CMAKE_INSTALL_PREFIX})
    ENDFOREACH()
    FOREACH(_task libtmg)
        LIST(APPEND TMG_ADD_INSTALL_STEP tmg_copy_${_task})
        SET(tmg_copy_${_task}_CMD ${CMAKE_COMMAND} -E copy
                                     ${TMG_BUILD_PATH}/${_task}.${MORSE_LIBRARY_EXTENSION} .)
        SET(tmg_copy_${_task}_DIR ${TMG_PATH}/lib)
    ENDFOREACH()

    # Install the external package
    # ----------------------------
    INSTALL_EXTERNAL_PACKAGE("tmg" "${TMG_USED_MODE}")

    # Set linker flags
    # ----------------
    SET(TMG_VENDOR       "tmg")
    SET(TMG_LIBRARY_PATH "${TMG_BUILD_PATH}")
    SET(TMG_LIBRARY      "${TMG_LIBRARY_PATH}/libtmg.${MORSE_LIBRARY_EXTENSION}")
    SET(TMG_LDFLAGS      "-L${TMG_LIBRARY_PATH} -ltmg")
    SET(TMG_LIBRARIES    "tmg")

ENDMACRO(INSTALL_TMG)

##
## @end file installTMG.cmake
##
