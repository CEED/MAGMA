###
#
#  @file infoLAPACKE.cmake
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
MACRO(LAPACKE_INFO_DEPS)
    # Define the direct dependencies of the library
    # ---------------------------------------------
    SET(LAPACKE_DIRECT_DEPS     "LAPACK" )

    # Define the priority of dependencies
    # -----------------------------------
    SET(LAPACKE_LAPACK_PRIORITY "depends")
    
ENDMACRO(LAPACKE_INFO_DEPS)

###
#   
#   
#   
###     
MACRO(LAPACKE_INFO_INSTALL)
    # Define web link of lapacke
    # --------------------------
    IF(NOT DEFINED LAPACKE_URL)
        SET(LAPACKE_URL "http://www.netlib.org/lapack/lapack.tgz")
    ENDIF()

    # Define tarball of lapacke
    # -------------------------
    IF(NOT DEFINED LAPACKE_TARBALL)
        SET(LAPACKE_TARBALL "lapack.tgz")
    ENDIF()

    # Define md5sum of lapacke
    # ------------------------
    IF(DEFINED LAPACKE_URL OR DEFINED LAPACKE_TARBALL)
        SET(LAPACKE_MD5SUM "61bf1a8a4469d4bdb7604f5897179478")
    ENDIF()

    # Define repository of lapacke
    # ----------------------------
    IF(NOT DEFINED LAPACKE_REPO_URL)
        SET(LAPACKE_REPO_MODE "SVN"                                               )
        SET(LAPACKE_REPO_URL  "https://icl.cs.utk.edu/svn/lapack-dev/lapack/trunk")
        SET(LAPACKE_REPO_ID   ""                                                  )
        SET(LAPACKE_REPO_PWD  ""                                                  )
    ENDIF()

ENDMACRO(LAPACKE_INFO_INSTALL)

###
#   
#   
#   
###     
MACRO(LAPACKE_INFO_FIND)
    # Define parameters for FIND_MY_PACKAGE
    # -------------------------------------
    SET(LAPACKE_type_library        "C;Fortran"                           )
    SET(LAPACKE_name_library        "lapacke"                             )
    SET(LAPACKE_name_pkgconfig      "LAPACKE_name_pkgconfig-NOTFOUND"     )
    SET(LAPACKE_name_include        "lapacke.h"                           )
    SET(LAPACKE_name_include_suffix "LAPACKE_name_include_suffix-NOTFOUND")
    SET(LAPACKE_name_binary         "LAPACKE_name_binary-NOTFOUND"        )
    IF(BUILD_SINGLE)
        LIST(APPEND LAPACKE_name_fct_test "LAPACKE_ssyev"                 )
    ENDIF()
    IF(BUILD_DOUBLE)
        LIST(APPEND LAPACKE_name_fct_test "LAPACKE_dsyev"                 )
    ENDIF()
    IF(BUILD_COMPLEX)
        LIST(APPEND LAPACKE_name_fct_test "LAPACKE_cheev"                 )
    ENDIF()
    IF(BUILD_COMPLEX16)
        LIST(APPEND LAPACKE_name_fct_test "LAPACKE_zheev"                 )
    ENDIF()

ENDMACRO(LAPACKE_INFO_FIND)

##
## @end file infoLAPACKE.cmake
##
