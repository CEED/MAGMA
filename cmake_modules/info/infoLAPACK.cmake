###
#
#  @file infoLAPACK.cmake
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
MACRO(LAPACK_INFO_DEPS)
    # Define the direct dependencies of the library
    # ---------------------------------------------
    SET(LAPACK_DIRECT_DEPS   "BLAS"   )

    # Define the priority of dependencies
    # -----------------------------------
    SET(LAPACK_BLAS_PRIORITY "depends")
    
ENDMACRO(LAPACK_INFO_DEPS)

###
#   
#   
#   
###     
MACRO(LAPACK_INFO_INSTALL)
    # Define web link of lapack
    # -------------------------
    IF(NOT DEFINED LAPACK_URL)
        SET(LAPACK_URL "http://www.netlib.org/lapack/lapack.tgz")
    ENDIF()

    # Define tarball of lapack
    # ------------------------
    IF(NOT DEFINED LAPACK_TARBALL)
        SET(LAPACK_TARBALL "lapack.tgz")
    ENDIF()

    # Define md5sum of lapack
    # -----------------------
    IF(DEFINED LAPACK_URL OR DEFINED LAPACK_TARBALL)
        SET(LAPACK_MD5SUM "61bf1a8a4469d4bdb7604f5897179478")
    ENDIF()

    # Define repository of lapack
    # ---------------------------
    IF(NOT DEFINED LAPACK_REPO_URL)
        SET(LAPACK_REPO_MODE "SVN"                                               )
        SET(LAPACK_REPO_URL  "https://icl.cs.utk.edu/svn/lapack-dev/lapack/trunk")
        SET(LAPACK_REPO_ID   ""                                                  )
        SET(LAPACK_REPO_PWD  ""                                                  )
    ENDIF()

ENDMACRO(LAPACK_INFO_INSTALL)

###
#   
#   
#   
###     
MACRO(LAPACK_INFO_FIND)
    # Define parameters for FIND_MY_PACKAGE
    # -------------------------------------
    SET(LAPACK_type_library        "C;Fortran"                          )
    SET(LAPACK_name_library        "LAPACK_name_library-NOTFOUND"       )
    SET(LAPACK_name_pkgconfig      "LAPACK_name_pkgconfig-NOTFOUND"     )
    SET(LAPACK_name_include        "LAPACK_name_include-NOTFOUND"       )
    SET(LAPACK_name_include_suffix "LAPACK_name_include_suffix-NOTFOUND")
    SET(LAPACK_name_binary         "LAPACK_name_binary-NOTFOUND"        )
    IF(BUILD_SINGLE)
        LIST(APPEND LAPACK_name_fct_test "ssyev"                        )
    ENDIF()
    IF(BUILD_DOUBLE)
        LIST(APPEND LAPACK_name_fct_test "dsyev"                        )
    ENDIF()
    IF(BUILD_COMPLEX)
        LIST(APPEND LAPACK_name_fct_test "cheev"                        )
    ENDIF()
    IF(BUILD_COMPLEX16)
        LIST(APPEND LAPACK_name_fct_test "zheev"                        )
    ENDIF()

ENDMACRO(LAPACK_INFO_FIND)

##
## @end file infoLAPACK.cmake
##
