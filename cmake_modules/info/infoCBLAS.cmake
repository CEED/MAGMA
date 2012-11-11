###
#
#  @file infoCBLAS.cmake
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
MACRO(CBLAS_INFO_DEPS)
    # Define the direct dependencies of the library
    # ---------------------------------------------
    SET(CBLAS_DIRECT_DEPS   "BLAS"   )

    # Define the priority of dependencies
    # -----------------------------------
    SET(CBLAS_BLAS_PRIORITY "depends")

ENDMACRO(CBLAS_INFO_DEPS)

###
#
#
#
###
MACRO(CBLAS_INFO_INSTALL)

    # Define web link of cblas
    # ------------------------
    IF(NOT DEFINED CBLAS_URL)
        SET(CBLAS_URL     "http://www.netlib.org/blas/blast-forum/cblas.tgz")
    ENDIF()

    # Define tarball of cblas
    # -----------------------
    IF(NOT DEFINED CBLAS_TARBALL)
        SET(CBLAS_TARBALL "cblas.tgz"                       )
    ENDIF()

    # Define md5sum of cblas
    # ----------------------
    IF(DEFINED CBLAS_URL OR DEFINED CBLAS_TARBALL)
        SET(CBLAS_MD5SUM  "1e8830f622d2112239a4a8a83b84209a")
    ENDIF()

    # Define repository of cblas
    # --------------------------
    IF(NOT DEFINED CBLAS_REPO_URL)
        SET(CBLAS_REPO_MODE "SVN")
        SET(CBLAS_REPO_URL   ""   )
        SET(CBLAS_REPO_ID    ""   )
        SET(CBLAS_REPO_PWD   ""   )
    ENDIF()

ENDMACRO(CBLAS_INFO_INSTALL)

###
#
#
#
###
MACRO(CBLAS_INFO_FIND)
    # Define parameters for FIND_MY_PACKAGE
    # -------------------------------------
    UNSET(CBLAS_name_fct_test)
    SET(CBLAS_type_library        "C"                                 )
    SET(CBLAS_name_library        "cblas"                             )
    SET(CBLAS_name_include        "CBLAS_name_include-NOTFOUND"       )
    SET(CBLAS_name_pkgconfig      "CBLAS_name_pkgconfig-NOTFOUND"     )
    SET(CBLAS_name_include_suffix "CBLAS_name_include_suffix-NOTFOUND")
    SET(CBLAS_name_binary         "CBLAS_name_binary-NOTFOUND"        )
    IF(BUILD_SINGLE)
        LIST(APPEND CBLAS_name_fct_test "cblas_sgemm"                 )
    ENDIF()
    IF(BUILD_DOUBLE)
        LIST(APPEND CBLAS_name_fct_test "cblas_dgemm"                 )
    ENDIF()
    IF(BUILD_COMPLEX)
        LIST(APPEND CBLAS_name_fct_test "cblas_cgemm"                 )
    ENDIF()
    IF(BUILD_COMPLEX16)
        LIST(APPEND CBLAS_name_fct_test "cblas_zgemm"                 )
    ENDIF()

ENDMACRO(CBLAS_INFO_FIND)

##
## @end file infoCBLAS.cmake
##
