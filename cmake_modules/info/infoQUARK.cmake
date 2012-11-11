###
#
#  @file infoQUARK.cmake 
#
#  @project QUARK
#  QUARK is a software package provided by:
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
MACRO(QUARK_INFO_DEPS)
    # Define the direct dependencies of the library
    # ---------------------------------------------
    SET(QUARK_DIRECT_DEPS    "HWLOC"  )
    
    # Define the priority of dependencies
    # -----------------------------------
    SET(QUARK_HWLOC_PRIORITY "depends")

ENDMACRO(QUARK_INFO_DEPS)

###
#   
#   
#   
###     
MACRO(QUARK_INFO_INSTALL)
    # Define web link of plasma
    # -------------------------
    IF(NOT DEFINED QUARK_URL)
        SET(QUARK_URL     "http://icl.cs.utk.edu/projectsfiles/quark/pubs/quark-0.9.0.tgz") 
    ENDIF()

    # Define tarball of plasma
    # ------------------------
    IF(NOT DEFINED QUARK_TARBALL)
        SET(QUARK_TARBALL "quark-0.9.0.tgz")
    ENDIF()

    # Define md5sum of plasma
    # -----------------------
    IF(DEFINED QUARK_URL OR DEFINED QUARK_TARBALL)
        SET(QUARK_MD5SUM  "52066a24b21c390d2f4fb3b57e976d08")
    ENDIF()

    # Define repository of plasma
    # ---------------------------
    IF(NOT DEFINED QUARK_REPO_URL)
        SET(QUARK_REPO_MODE "SVN")
        SET(QUARK_REPO_URL   ""   )
        SET(QUARK_REPO_ID    ""   )
        SET(QUARK_REPO_PWD   ""   )
    ENDIF()

ENDMACRO(QUARK_INFO_INSTALL)

###
#   
#   
#   
###     
MACRO(QUARK_INFO_FIND)
    # Define parameters for FIND_MY_PACKAGE
    # -------------------------------------
    SET(QUARK_type_library        "C"                                 )
    SET(QUARK_name_library        "quark"                             )
    SET(QUARK_name_pkgconfig      "QUARK_name_pkgconfig-NOTFOUND"     )
    SET(QUARK_name_include        "quark.h"                           )
    SET(QUARK_name_include_suffix "QUARK_name_include_suffix-NOTFOUND")
    SET(QUARK_name_binary         "QUARK_name_binary-NOTFOUND"        )
    SET(QUARK_name_fct_test       "quark_setaffinity"                 )

ENDMACRO(QUARK_INFO_FIND)

##
## @end file infoQUARK.cmake 
##
