###
#
#  @file infoEIGEN.cmake
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
MACRO(EIGEN_INFO_DEPS)
    # Define the direct dependencies of the library
    # ---------------------------------------------
    SET(EIGEN_DIRECT_DEPS "")

ENDMACRO(EIGEN_INFO_DEPS)

###
#
#
#
###
MACRO(EIGEN_INFO_INSTALL)
    # Define web link of eigen
    # ------------------------
    IF(NOT DEFINED EIGEN_URL)
        SET(EIGEN_URL     "http://bitbucket.org/eigen/eigen/get/3.1.1.tar.gz")
    ENDIF()

    # Define tarball of eigen
    # -----------------------
    IF(NOT DEFINED EIGEN_TARBALL)
        SET(EIGEN_TARBALL "3.1.1.tar.gz")
    ENDIF()

    # Define md5sum of eigen
    # ----------------------
    IF(DEFINED EIGEN_URL OR DEFINED EIGEN_TARBALL)
        SET(EIGEN_MD5SUM  "7f1de87d4bfef65d0c59f15f6697829d")
    ENDIF()

    # Define repository of eigen
    # --------------------------
    IF(NOT DEFINED EIGEN_REPO_URL)
        SET(EIGEN_REPO_MODE  "")
        SET(EIGEN_REPO_URL   "")
        SET(EIGEN_REPO_ID    "")
        SET(EIGEN_REPO_PWD   "")
    ENDIF()

ENDMACRO(EIGEN_INFO_INSTALL)

###
#
#
#
###
MACRO(EIGEN_INFO_FIND)
    # Define parameters for FIND_MY_PACKAGE
    # -------------------------------------
    SET(EIGEN_type_library        "BLAS_type_library-NOTFOUND"       )
    SET(EIGEN_name_library        "BLAS_name_library-NOTFOUND"       )
    SET(EIGEN_name_pkgconfig      "BLAS_name_pkgconfig-NOTFOUND"     )
    SET(EIGEN_name_include        "BLAS_name_include-NOTFOUND"       )
    SET(EIGEN_name_include_suffix "BLAS_name_include_suffix-NOTFOUND")
    SET(EIGEN_name_binary         "BLAS_name_binary-NOTFOUND"        )

    UNSET(EIGEN_name_fct_test)
    IF(BUILD_SINGLE)
        LIST(APPEND EIGEN_name_fct_test "sgemm")
    ENDIF()
    IF(BUILD_DOUBLE)
        LIST(APPEND EIGEN_name_fct_test "dgemm")
    ENDIF()
    IF(BUILD_COMPLEX)
        LIST(APPEND EIGEN_name_fct_test "cgemm")
    ENDIF()
    IF(BUILD_COMPLEX16)
        LIST(APPEND EIGEN_name_fct_test "zgemm")
    ENDIF()


ENDMACRO(EIGEN_INFO_FIND)

##
## @end file infoEIGEN.cmake
##
