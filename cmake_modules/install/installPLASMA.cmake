###
#
#  @file installPLASMA.cmake
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

MACRO(INSTALL_PLASMA _MODE)

    # Define prefix paths
    # -------------------
    IF(MORSE_SEPARATE_PROJECTS)
        SET(PLASMA_PATH ${CMAKE_INSTALL_PREFIX}/plasma)
    ELSE(MORSE_SEPARATE_PROJECTS)
        SET(PLASMA_PATH ${CMAKE_INSTALL_PREFIX})
    ENDIF(MORSE_SEPARATE_PROJECTS)

    # Create make.inc
    # ---------------
    IF(EXISTS ${CMAKE_BINARY_DIR}/plasma_make.inc)
        FILE(REMOVE ${CMAKE_BINARY_DIR}/plasma_make.inc)
    ENDIF()
    STRING(TOUPPER "${CMAKE_BUILD_TYPE}" TYPE)
    FILE(APPEND ${CMAKE_BINARY_DIR}/plasma_make.inc
"
prefix    ?= ${PLASMA_PATH}
CC         = ${CMAKE_C_COMPILER}
FC         = ${CMAKE_Fortran_COMPILER}
LOADER     = ${CMAKE_Fortran_COMPILER}

ARCH       = ${CMAKE_AR}
ARCHFLAGS  = cr
RANLIB     = ${CMAKE_RANLIB}

CFLAGS     = ${CMAKE_C_FLAGS_${TYPE}}
CFLAGS    += ${CMAKE_C_FLAGS}
CFLAGS    += ${FORTRAN_MANGLING_DETECTED}
CFLAGS    += -I${QUARK_INCLUDE_PATH}

FFLAGS     = ${CMAKE_Fortran_FLAGS_${TYPE}}
FFLAGS    += ${CMAKE_Fortran_FLAGS}
FFLAGS    += ${CMAKE_Fortran_LDFLAGS} 

LDFLAGS   += ${CMAKE_Fortran_LDFLAGS} 
LDFLAGS   += ${FORTRAN_MANGLING_DETECTED}
LDFLAGS   += ${QUARK_LDFLAGS}

LIBBLAS    = ${BLAS_LDFLAGS}
LIBCBLAS   = ${CBLAS_LDFLAGS}
LIBLAPACK  = ${LAPACK_LDFLAGS}
INCCLAPACK = -I${LAPACKE_INCLUDE_PATH}
LIBCLAPACK = ${LAPACKE_LDFLAGS}
")

    IF(HAVE_QUARK)
        FILE(APPEND ${CMAKE_BINARY_DIR}/plasma_make.inc
"
QUARKDIR  = ${QUARK_PATH}
")
    ENDIF(HAVE_QUARK)

    IF(BUILD_SHARED_LIBS)
        FILE(APPEND ${CMAKE_BINARY_DIR}/plasma_make.inc "
CFLAGS    += -fPIC
FFLAGS    += -fPIC
ARCHFLAGS  = rcs
")
    ENDIF(BUILD_SHARED_LIBS)

    # Define steps of installation
    # ----------------------------
    SET(PLASMA_SOURCE_PATH ${CMAKE_BINARY_DIR}/externals/plasma)
    SET(PLASMA_BUILD_PATH  ${CMAKE_BINARY_DIR}/externals/plasma)
    SET(PLASMA_PATCH_CMD   patch -p1 -i ${CMAKE_SOURCE_DIR}/patch/plasma_freebsd.patch)
    SET(PLASMA_CONFIG_CMD  ${CMAKE_COMMAND} -E copy
                             ${CMAKE_BINARY_DIR}/plasma_make.inc
                             ${PLASMA_BUILD_PATH}/make.inc)
    SET(PLASMA_BUILD_CMD   ${CMAKE_MAKE_PROGRAM} libplasma libcoreblas)
    SET(PLASMA_INSTALL_CMD ${CMAKE_MAKE_PROGRAM} install)

    # Install the external package
    # ----------------------------
    INSTALL_EXTERNAL_PACKAGE("plasma" "${PLASMA_USED_MODE}")

    # Set linker flags
    # ----------------
    SET(PLASMA_LIBRARY_PATH    "${PLASMA_SOURCE_PATH}/lib"    )
    SET(PLASMA_INCLUDE_PATH    "${PLASMA_SOURCE_PATH}/include")
    SET(PLASMA_LIBRARY         "${PLASMA_LIBRARY_PATH}/libplasma.${MORSE_LIBRARY_EXTENSION}"  )
    LIST(APPEND PLASMA_LIBRARY "${PLASMA_LIBRARY_PATH}/libcoreblas.${MORSE_LIBRARY_EXTENSION}")
    LIST(APPEND PLASMA_LIBRARY "${PLASMA_LIBRARY_PATH}/libplasma.${MORSE_LIBRARY_EXTENSION}"  )
    SET(PLASMA_LIBRARIES       "plasma;coreblas;plasma")
    SET(PLASMA_LDFLAGS         "-L${PLASMA_LIBRARY_PATH} -lplasma -lcoreblas -lplasma")

ENDMACRO(INSTALL_PLASMA)

##
## @end file installPLASMA.cmake
##
