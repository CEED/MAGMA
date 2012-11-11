###
#
#  @file infoOPENMPI.cmake
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
MACRO(MPI_INFO_DEPS)
    # Define the direct dependencies of the library
    # ---------------------------------------------
    SET(MPI_DIRECT_DEPS "") 

ENDMACRO(MPI_INFO_DEPS)

###
#   
#   
#   
###     
MACRO(MPI_INFO_INSTALL)
    # Define web link of openmpi
    # --------------------------
    IF(NOT DEFINED MPI_URL)
        SET(MPI_URL "http://www.open-mpi.org/software/ompi/v1.6/downloads/openmpi-1.6.2.tar.gz")
    ENDIF()

    # Define tarball of openmpi
    # -------------------------
    IF(NOT DEFINED MPI_TARBALL)
        SET(MPI_TARBALL "openmpi-1.6.2.tar.gz")
    ENDIF()

    # Define md5sum of openmpi
    # ------------------------
    IF(DEFINED MPI_URL OR DEFINED MPI_TARBALL)
        SET(MPI_MD5SUM  "351845a0edd8141feb30d31edd59cdcd")
    ENDIF()

    # Define repository of openmpi
    # ----------------------------
    IF(NOT DEFINED MPI_REPO_URL)
        SET(MPI_REPO_MODE "SVN"                                   )
        SET(MPI_REPO_URL   "http://svn.open-mpi.org/svn/ompi/trunk")
        SET(MPI_REPO_ID    ""                                      )
        SET(MPI_REPO_PWD   ""                                      )
    ENDIF()

ENDMACRO(MPI_INFO_INSTALL)

###
#   
#   
#   
###     
MACRO(MPI_INFO_FIND)
    # Define parameters for FIND_MY_PACKAGE
    # -------------------------------------
    SET(MPI_type_library        "MPI_type_library-NOTFOUND"       )
    SET(MPI_name_library        "MPI_name_library-NOTFOUND"       )
    SET(MPI_name_pkgconfig      "MPI_name_pkgconfig-NOTFOUND"     )
    SET(MPI_name_include        "MPI_name_include-NOTFOUND"       )
    SET(MPI_name_include_suffix "MPI_name_include_suffix-NOTFOUND")
    SET(MPI_name_fct_test       "MPI_INIT"                        )
    SET(MPI_name_binary         "MPI_name_binary-NOTFOUND"        )

ENDMACRO(MPI_INFO_FIND)

##
## @end file infoOPENMPI.cmake
##
