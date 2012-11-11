###
#
# @file infoAPPML.cmake
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
MACRO(APPML_INFO_DEPS)
    # Define the direct dependencies of the library
    # ---------------------------------------------
    SET(APPML_DIRECT_DEPS   "OPENCL")

    # Define the priority of dependencies
    # -----------------------------------
    SET(APPML_OPENCL_PRIORITY "depends")

ENDMACRO(APPML_INFO_DEPS)

###
#
#
#
###
MACRO(APPML_INFO_INSTALL)

    # Define web link of appml
    # ------------------------
    IF(NOT DEFINED APPML_URL)
        SET(APPML_URL     "APPML_URL-NOTFOUND"   )
    ENDIF()

    # Define tarball of appml
    # -----------------------
    IF(NOT DEFINED APPML_TARBALL)
        SET(APPML_TARBALL "APPML_TARBALL-NOTFOUND")
    ENDIF()

    # Define md5sum of appml
    # ----------------------
    IF(DEFINED APPML_URL OR DEFINED APPML_TARBALL)
        SET(APPML_MD5SUM  "APPML_MD5SUM-NOTFOUND" )
    ENDIF()

    # Define repository of appml
    # --------------------------
    IF(NOT DEFINED APPML_REPO_URL)
        SET(APPML_REPO_MODE "APPML_REPO_MODE-NOTFOUND")
        SET(APPML_REPO_URL   ""                        )
        SET(APPML_REPO_ID    ""                        )
        SET(APPML_REPO_PWD   ""                        )
    ENDIF()

ENDMACRO(APPML_INFO_INSTALL)

###
#
#
#
###
MACRO(APPML_INFO_FIND)
    # Define parameters for FIND_MY_PACKAGE
    # -------------------------------------
    SET(APPML_type_library        "C"                                 )
    SET(APPML_name_library        "clAmdBlas"                         )
    SET(APPML_name_pkgconfig      "APPML_name_pkgconfig-NOTFOUND"     )
    SET(APPML_name_include        "clAmdBlas.h"                       )
    SET(APPML_name_include_suffix "APPML_name_include_suffix-NOTFOUND")
    SET(APPML_name_binary         "APPML_name_binary-NOTFOUND"        )
    IF(BUILD_SINGLE)
        SET(APPML_name_fct_test   "clAmdBlasSgemm"                    )
    ENDIF()
    IF(BUILD_DOUBLE)
        SET(APPML_name_fct_test   "clAmdBlasDgemm"                    )
    ENDIF()
    IF(BUILD_COMPLEX)
        SET(APPML_name_fct_test   "clAmdBlasCgemm"                    )
    ENDIF()
    IF(BUILD_COMPLEX16)
        SET(APPML_name_fct_test   "clAmdBlasZgemm"                    )
    ENDIF()

ENDMACRO(APPML_INFO_FIND)

###
### END infoAPPML.cmake
###
