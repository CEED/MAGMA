###
#
#  @file infoREFBLAS.cmake
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
MACRO(REFBLAS_INFO_DEPS)
    # Define the direct dependencies of the library
    # ---------------------------------------------
    SET(REFBLAS_DIRECT_DEPS "")

ENDMACRO(REFBLAS_INFO_DEPS)

###
#
#
#
###
MACRO(REFBLAS_INFO_INSTALL)

    # Define web link of blas
    # -----------------------
    IF(NOT DEFINED REFBLAS_URL)
        SET(REFBLAS_URL     "http://netlib.org/blas/blas.tgz")
    ENDIF()

    # Define tarball of blas
    # ----------------------
    IF(NOT DEFINED REFBLAS_TARBALL)
        SET(REFBLAS_TARBALL "blas.tgz")
    ENDIF()

    # Define md5sum of blas
    # ---------------------
    IF(DEFINED REFBLAS_URL OR DEFINED REFBLAS_TARBALL)
        SET(REFBLAS_MD5SUM  "5e99e975f7a1e3ea6abcad7c6e7e42e6")
    ENDIF()

    # Define repository of blas
    # -------------------------
    IF(NOT DEFINED REFBLAS_REPO_URL)
        SET(REFBLAS_REPO_MODE "SVN")
        SET(REFBLAS_REPO_URL   ""   )
        SET(REFBLAS_REPO_ID    ""   )
        SET(REFBLAS_REPO_PWD   ""   )
    ENDIF()

ENDMACRO(REFBLAS_INFO_INSTALL)

###
#   
#   
#   
###     
MACRO(REFBLAS_INFO_FIND)
    # Define parameters for FIND_MY_PACKAGE
    # -------------------------------------
    SET(REFBLAS_type_library        "Fortran"                          )
    SET(REFBLAS_name_library        "BLAS_name_library-NOTFOUND"       )
    SET(REFBLAS_name_pkgconfig      "BLAS_name_pkgconfig-NOTFOUND"     )
    SET(REFBLAS_name_include        "BLAS_name_include-NOTFOUND"       )
    SET(REFBLAS_name_include_suffix "BLAS_name_include_suffix-NOTFOUND")
    SET(REFBLAS_name_binary         "BLAS_name_binary-NOTFOUND"        )

    UNSET(REFBLAS_name_fct_test)
    IF(BUILD_SINGLE)
        LIST(APPEND REFBLAS_name_fct_test "sgemm")
    ENDIF()
    IF(BUILD_DOUBLE)
        LIST(APPEND REFBLAS_name_fct_test "dgemm")
    ENDIF()
    IF(BUILD_COMPLEX)
        LIST(APPEND REFBLAS_name_fct_test "cgemm")
    ENDIF()
    IF(BUILD_COMPLEX16)
        LIST(APPEND REFBLAS_name_fct_test "zgemm")
    ENDIF()


ENDMACRO(REFBLAS_INFO_FIND)

##
## @end file infoREFBLAS.cmake
##
