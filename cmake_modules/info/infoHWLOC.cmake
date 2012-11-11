###
#
#  @file infoHWLOC.cmake
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
MACRO(HWLOC_INFO_DEPS)
    # Define the direct dependencies of the library
    # ---------------------------------------------
    SET(HWLOC_DIRECT_DEPS "")
    
ENDMACRO(HWLOC_INFO_DEPS)

###
#   
#   
#   
###     
MACRO(HWLOC_INFO_INSTALL)
    # Define web link of hwloc
    # ------------------------
    IF(NOT DEFINED HWLOC_URL)
        SET(HWLOC_URL "http://www.open-mpi.org/software/hwloc/v1.5/downloads/hwloc-1.5.tar.gz")
    ENDIF()

    # Define tarball of hwloc
    # -----------------------
    IF(NOT DEFINED HWLOC_TARBALL)
        SET(HWLOC_TARBALL "hwloc-1.5.tar.gz")
    ENDIF()

    # Define md5sum of hwloc
    # ----------------------
    IF(DEFINED HWLOC_URL OR DEFINED HWLOC_TARBALL)
        SET(HWLOC_MD5SUM  "1fd9b3d864f4f3f74d0c49bda267c6ff")
    ENDIF()

    # Define repository of hwloc
    # --------------------------
    IF(NOT DEFINED HWLOC_REPO_URL)
        SET(HWLOC_REPO_MODE "SVN"                                     )
        SET(HWLOC_REPO_URL   "https://svn.open-mpi.org/svn/hwloc/trunk")
        SET(HWLOC_REPO_ID    ""                                        )
        SET(HWLOC_REPO_PWD   ""                                        )
    ENDIF()

ENDMACRO(HWLOC_INFO_INSTALL)

###
#   
#   
#   
###     
MACRO(HWLOC_INFO_FIND)
    # Define parameters for FIND_MY_PACKAGE
    # -------------------------------------
    SET(HWLOC_type_library        "C"                                 )
    SET(HWLOC_name_library        "hwloc"                             )
    SET(HWLOC_name_pkgconfig      "hwloc"                             )
    SET(HWLOC_name_include        "hwloc.h"                           )
    SET(HWLOC_name_include_suffix "HWLOC_name_include_suffix-NOTFOUND")
    SET(HWLOC_name_fct_test       "hwloc_set_cpubind"                 )
    SET(HWLOC_name_binary         "HWLOC_name_binary-NOTFOUND"        )
    #SET(HWLOC_name_binary        "hwloc-ls"                          )

ENDMACRO(HWLOC_INFO_FIND)

##
## @end file infoHWLOC.cmake
##
