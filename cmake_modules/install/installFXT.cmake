###
#
#  @file installFXT.cmake
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

MACRO(INSTALL_FXT _MODE)

    # Define prefix paths
    # -------------------
    IF(MORSE_SEPARATE_PROJECTS)
        SET(FXT_PATH ${CMAKE_INSTALL_PREFIX}/fxt)
    ELSE(MORSE_SEPARATE_PROJECTS)
        SET(FXT_PATH ${CMAKE_INSTALL_PREFIX})
    ENDIF(MORSE_SEPARATE_PROJECTS)

    # Define options
    # --------------
    UNSET(FXT_CONFIG_OPTS)
    LIST(APPEND FXT_CONFIG_OPTS --prefix=${FXT_PATH})
    LIST(APPEND FXT_CONFIG_OPTS --enable-shared)
    IF(NOT BUILD_SHARED_LIBS)
        LIST(APPEND FXT_CONFIG_OPTS --enable-static)
    ENDIF()

    # Define steps of installation
    # ----------------------------
    SET(FXT_SOURCE_PATH ${CMAKE_BINARY_DIR}/externals/fxt)
    SET(FXT_BUILD_PATH  ${CMAKE_BINARY_DIR}/externals/fxt)
    SET(FXT_CONFIG_CMD  ./configure)
    SET(FXT_BUILD_CMD   ${CMAKE_MAKE_PROGRAM})
    SET(FXT_INSTALL_CMD ${CMAKE_MAKE_PROGRAM} install)

    # Install the external package
    # -----------------------------
    INSTALL_EXTERNAL_PACKAGE("fxt" "${FXT_USED_MODE}")

    # Set linker flags
    # ----------------
    SET(FXT_BINARY_PATH  "${FXT_BUILD_PATH}/tools/.libs")
    SET(FXT_LIBRARY_PATH "${FXT_BUILD_PATH}/tools/.libs")
    SET(FXT_INCLUDE_PATH "${FXT_BUILD_PATH}/tools")
    SET(FXT_LIBRARY      "${FXT_LIBRARY_PATH}/libfxt.${MORSE_LIBRARY_EXTENSION}")
    SET(FXT_LDFLAGS      "-L${FXT_LIBRARY_PATH} -lfxt")
    SET(FXT_LIBRARIES    "fxt")

ENDMACRO(INSTALL_FXT)

##
## @end file installFXT.cmake
##

