###
#
#  @file installQUARK.cmake
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

MACRO(INSTALL_QUARK _MODE)

    # Define prefix paths
    # -------------------
    IF(MORSE_SEPARATE_PROJECTS)
        SET(QUARK_PATH ${CMAKE_INSTALL_PREFIX}/quark)
    ELSE(MORSE_SEPARATE_PROJECTS)
        SET(QUARK_PATH ${CMAKE_INSTALL_PREFIX})
    ENDIF(MORSE_SEPARATE_PROJECTS)

    # Create make.inc
    # ---------------
    IF(EXISTS ${CMAKE_BINARY_DIR}/quark_make.inc)
        FILE(REMOVE ${CMAKE_BINARY_DIR}/quark_make.inc)
    ENDIF()
    STRING(TOUPPER "${CMAKE_BUILD_TYPE}" TYPE)
    FILE(APPEND ${CMAKE_BINARY_DIR}/quark_make.inc
"
prefix  ?= ${QUARK_PATH}
CC       = ${CMAKE_C_COMPILER}
CFLAGS   = ${CMAKE_C_FLAGS_${TYPE}}
CFLAGS  += ${CMAKE_C_FLAGS}
AR       = ${CMAKE_AR}
ARFLAGS  = cr
RANLIB   = ${CMAKE_RANLIB}
LD       = $(CC)
")

    IF(HAVE_PTHREAD)
        FILE(APPEND ${CMAKE_BINARY_DIR}/quark_make.inc
"
LDLIBS   = ${CMAKE_THREAD_LIBS_INIT}
")
    ENDIF(HAVE_PTHREAD)

    IF(HAVE_HWLOC)
        FILE(APPEND ${CMAKE_BINARY_DIR}/quark_make.inc 
"
CFLAGS  += -DQUARK_HWLOC
CFLAGS  += -I${HWLOC_INCLUDE_PATH}
LD      += ${HWLOC_LDFLAGS}
")

    ENDIF(HAVE_HWLOC)

    IF(BUILD_SHARED_LIBS)
        FILE(APPEND ${CMAKE_BINARY_DIR}/quark_make.inc
"
CFLAGS    += -fPIC
ARCHFLAGS  = rcs
")
    ENDIF(BUILD_SHARED_LIBS)

    FILE(APPEND ${CMAKE_BINARY_DIR}/quark_make.inc
"
.c.o:
\t$(CC) $(CFLAGS) -c $< -o $@
")

    # Define steps of installation
    # ----------------------------
    SET(QUARK_SOURCE_PATH ${CMAKE_BINARY_DIR}/externals/quark)
    SET(QUARK_BUILD_PATH  ${CMAKE_BINARY_DIR}/externals/quark)
    SET(QUARK_PATCH_CMD   patch -p1 -i ${CMAKE_SOURCE_DIR}/patch/quark_freebsd.patch)
    SET(QUARK_CONFIG_CMD  ${CMAKE_COMMAND} -E copy
                            ${CMAKE_BINARY_DIR}/quark_make.inc
                            ${QUARK_BUILD_PATH}/make.inc)
    SET(QUARK_BUILD_CMD   ${CMAKE_MAKE_PROGRAM} all)
    SET(QUARK_INSTALL_CMD ${CMAKE_MAKE_PROGRAM} install)

    # Define additional step
    # ----------------------
    UNSET(QUARK_ADD_INSTALL_STEP)
    FOREACH(_task icl_hash icl_list)
        LIST(APPEND QUARK_ADD_INSTALL_STEP quark_copy_${_task}_h)
        SET(quark_copy_${_task}_h_CMD ${CMAKE_COMMAND} -E copy ${QUARK_BUILD_PATH}/${_task}.h .)
        SET(quark_copy_${_task}_h_DIR ${QUARK_PATH}/include)
    ENDFOREACH()

    # Install the external package
    # ----------------------------
    INSTALL_EXTERNAL_PACKAGE("quark" "${QUARK_USED_MODE}")

    # Set linker flags
    # ----------------
    SET(QUARK_LIBRARY_PATH "${QUARK_BUILD_PATH}")
    SET(QUARK_INCLUDE_PATH "${QUARK_BUILD_PATH}")
    SET(QUARK_LIBRARY      "${QUARK_LIBRARY_PATH}/libquark.${MORSE_LIBRARY_EXTENSION}")
    SET(QUARK_LDFLAGS      "-L${QUARK_LIBRARY_PATH} -lquark")
    SET(QUARK_LIBRARIES    "quark")

ENDMACRO(INSTALL_QUARK)

##
## @end file installQUARK.cmake
##
