###
#
#  @file installLAPACK.cmake
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

MACRO(INSTALL_LAPACK _MODE)

    # Define prefix paths
    # -------------------
    IF(MORSE_SEPARATE_PROJECTS)
        SET(LAPACK_PATH ${CMAKE_INSTALL_PREFIX}/lapack)
    ELSE(MORSE_SEPARATE_PROJECTS)
        SET(LAPACK_PATH ${CMAKE_INSTALL_PREFIX})
    ENDIF(MORSE_SEPARATE_PROJECTS)

    # Create make.inc
    # ---------------
    IF(EXISTS ${CMAKE_BINARY_DIR}/lapack_make.inc)
        FILE(REMOVE ${CMAKE_BINARY_DIR}/lapack_make.inc)
    ENDIF()
    STRING(TOUPPER "${CMAKE_BUILD_TYPE}" TYPE)
    FILE(APPEND ${CMAKE_BINARY_DIR}/lapack_make.inc
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
        FILE(APPEND ${CMAKE_BINARY_DIR}/lapack_make.inc "
OPTS      += -fPIC
NOOPT     += -fPIC
DRVOPTS   += -fPIC
ARCHFLAGS  = rcs
")
    ENDIF(BUILD_SHARED_LIBS)

    # Define steps of installation
    # ----------------------------
    SET(LAPACK_SOURCE_PATH ${CMAKE_BINARY_DIR}/externals/lapack)
    SET(LAPACK_BUILD_PATH  ${CMAKE_BINARY_DIR}/externals/lapack)
    SET(LAPACK_SOURCE_PATH ${CMAKE_BINARY_DIR}/externals/lapack)
    SET(LAPACK_CONFIG_CMD  ${CMAKE_COMMAND} -E copy
                             ${CMAKE_BINARY_DIR}/lapack_make.inc
                             ${LAPACK_BUILD_PATH}/make.inc)
    SET(LAPACK_BUILD_CMD     ${CMAKE_MAKE_PROGRAM} lapacklib)
    SET(LAPACK_INSTALL_CMD ${CMAKE_COMMAND} -E make_directory ${LAPACK_PATH})

    # Define additional step
    # ----------------------
    UNSET(LAPACK_ADD_INSTALL_STEP)
    FOREACH(_task lib)
        LIST(APPEND LAPACK_ADD_INSTALL_STEP lapack_create_${_task}_path)
        SET(lapack_create_${_task}_path_CMD ${CMAKE_COMMAND} -E make_directory ${LAPACK_PATH}/${_task})
        SET(lapack_create_${_task}_path_DIR ${CMAKE_INSTALL_PREFIX})
    ENDFOREACH()
    FOREACH(_task liblapack)
        LIST(APPEND LAPACK_ADD_INSTALL_STEP lapack_copy_${_task})
        SET(lapack_copy_${_task}_CMD ${CMAKE_COMMAND} -E copy
                                     ${LAPACK_BUILD_PATH}/${_task}.${MORSE_LIBRARY_EXTENSION} .)
        SET(lapack_copy_${_task}_DIR ${LAPACK_PATH}/lib)
    ENDFOREACH()

    # Install the external package
    # ----------------------------
    INSTALL_EXTERNAL_PACKAGE("lapack" "${LAPACK_USED_MODE}")

    # Set linker flags
    # ----------------
    SET(LAPACK_VENDOR       "lapack")
    SET(LAPACK_LIBRARY_PATH "${LAPACK_BUILD_PATH}")
    SET(LAPACK_LIBRARY      "${LAPACK_LIBRARY_PATH}/liblapack.${MORSE_LIBRARY_EXTENSION}")
    SET(LAPACK_LDFLAGS      "-L${LAPACK_LIBRARY_PATH} -llapack")
    SET(LAPACK_LIBRARIES    "lapack")

    STRING(TOUPPER "${BLAS_VENDOR}" VALUE_MORSE_USE_BLAS)
    IF("${VALUE_MORSE_USE_BLAS}" MATCHES "EIGEN")
        SET(LAPACK_VENDOR       "eigen+lapack")
        SET(LAPACK_LIBRARY_PATH "${EIGENLAPACK_LIBRARY_PATH};${LAPACK_LIBRARY_PATH}")
        SET(LAPACK_LIBRARY      "${EIGENLAPACK_LIBRARY};${LAPACK_LIBRARY}")
        SET(LAPACK_LDFLAGS      "${EIGENLAPACK_LDFLAGS} ${LAPACK_LDFLAGS}")
        SET(LAPACK_LIBRARIES    "${EIGENLAPACK_LIBRARIES};${LAPACK_LIBRARIES}")
    ENDIF()
    LIST(REMOVE_DUPLICATES LAPACK_LIBRARY_PATH)

ENDMACRO(INSTALL_LAPACK)

##
## @end file installLAPACK.cmake
##
