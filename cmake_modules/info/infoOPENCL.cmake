###
#
# @file infoOPENCL.cmake
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

###
#
#
#
###
MACRO(OPENCL_INFO_DEPS)
    # Define the direct dependencies of the library
    # ---------------------------------------------
    SET(OPENCL_DIRECT_DEPS   "")

ENDMACRO(OPENCL_INFO_DEPS)

###
#
#
#
###
MACRO(OPENCL_INFO_INSTALL)

    # Define web link of opencl
    # -------------------------
    IF(NOT DEFINED OPENCL_URL)
        SET(OPENCL_URL     "OPENCL_URL-NOTFOUND"   )
    ENDIF()

    # Define tarball of opencl
    # ------------------------
    IF(NOT DEFINED OPENCL_TARBALL)
        SET(OPENCL_TARBALL "OPENCL_TARBALL-NOTFOUND")
    ENDIF()

    # Define md5sum of opencl
    # -----------------------
    IF(DEFINED OPENCL_URL OR DEFINED OPENCL_TARBALL)
        SET(OPENCL_MD5SUM  "OPENCL_MD5SUM-NOTFOUND" )
    ENDIF()

    # Define repository of opencl
    # ---------------------------
    IF(NOT DEFINED OPENCL_REPO_URL)
        SET(OPENCL_REPO_MODE "OPENCL_REPO_MODE-NOTFOUND")
        SET(OPENCL_REPO_URL   ""                         )
        SET(OPENCL_REPO_ID    ""                         )
        SET(OPENCL_REPO_PWD   ""                         )
    ENDIF()

ENDMACRO(OPENCL_INFO_INSTALL)

###
#
#
#
###
MACRO(OPENCL_INFO_FIND)
    # Define parameters for FIND_MY_PACKAGE
    # -------------------------------------
    SET(OPENCL_type_library            "C"                             )
    SET(OPENCL_name_library            "OpenCL"                        )
    SET(OPENCL_name_pkgconfig          "OPENCL_name_pkgconfig-NOTFOUND")
    SET(OPENCL_name_include            "cl.h"                          )
    IF(APPLE)
        SET(OPENCL_name_include_suffix "OpenCL"                        )
    ELSE(APPLE)
        SET(OPENCL_name_include_suffix "CL"                            )
    ENDIF(APPLE)
    SET(OPENCL_name_fct_test           "clFinish"                      )
    SET(OPENCL_name_binary             "OPENCL_name_binary-NOTFOUND"   )

ENDMACRO(OPENCL_INFO_FIND)

##
## @end file infoOPENCL.cmake
##
