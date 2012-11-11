###
#
#  @file infoSTARPU.cmake
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
MACRO(STARPU_INFO_DEPS)
    # Define the direct dependencies of the library
    # ---------------------------------------------
    SET(STARPU_DIRECT_DEPS         "HWLOC" )
    LIST(APPEND STARPU_DIRECT_DEPS "CUDA"  )
    #LIST(APPEND STARPU_DIRECT_DEPS "OPENCL")
    LIST(APPEND STARPU_DIRECT_DEPS "MPI"   )
    LIST(APPEND STARPU_DIRECT_DEPS "FXT"   )

    # Define the priority of dependencies
    # -----------------------------------
    INCLUDE(BuildSystemTools)
    DEFINE_PRIORITY("STARPU" "HWLOC"  "depends" "depends"   "depends"    )
    DEFINE_PRIORITY("STARPU" "CUDA"   "depends" "suggests"  "recommends" )
    #DEFINE_PRIORITY("STARPU" "OPENCL" "depends" "suggests"  "recommends")
    DEFINE_PRIORITY("STARPU" "MPI"    "depends" "suggests"  "suggests"   )
    DEFINE_PRIORITY("STARPU" "FXT"    "depends" "suggests"  "suggests"   )

ENDMACRO(STARPU_INFO_DEPS)

###
#   
#   
#   
###     
MACRO(STARPU_INFO_INSTALL)
    # Define web link of starpu
    # -------------------------
    IF(NOT DEFINED STARPU_URL)
        SET(STARPU_URL     "http://runtime.bordeaux.inria.fr/StarPU/files/starpu-1.0.4.tar.gz")
    ENDIF()

    # Define tarball of starpu
    # ------------------------
    IF(NOT DEFINED STARPU_TARBALL)
        SET(STARPU_TARBALL "starpu-1.0.4.tar.gz")
    ENDIF()

    # Define md5sum of eigen
    # ----------------------
    IF(DEFINED STARPU_URL OR DEFINED STARPU_TARBALL)
        SET(STARPU_MD5SUM  "3954c0675ead43398cadb73cbcffd8e4")
    ENDIF()

    # Define repository of starpu
    # ---------------------------
    IF(NOT DEFINED STARPU_REPO_URL)
        SET(STARPU_REPO_MODE "SVN"                                    )
        SET(STARPU_REPO_URL   "https://scm.gforge.inria.fr/svn/starpu")
        SET(STARPU_REPO_ID    "anonsvn"                               )
        SET(STARPU_REPO_PWD   "anonsvn"                               )
    ENDIF()

ENDMACRO(STARPU_INFO_INSTALL)

###
#   
#   
#   
###     
MACRO(STARPU_INFO_FIND)
    # Define parameters for FIND_MY_PACKAGE
    # -------------------------------------
    SET(STARPU_type_library              "C"                          )
    SET(STARPU_name_fct_test             "starpu_task_get_current"    )
    SET(STARPU_name_include              "starpu.h;starpu_profiling.h")
    SET(STARPU_name_include_suffix       "starpu/1.0"                 )
    SET(STARPU_name_binary               "STARPU_name_binary-NOTFOUND")
    #SET(STARPU_name_binary               "starpu_calibrate_bus"       )

    IF(HAVE_MPI)
        SET(STARPU_name_library          "starpumpi-1.0;starpu-1.0"   )
        SET(STARPU_name_pkgconfig        "starpumpi-1.0"              )
        LIST(APPEND STARPU_name_fct_test "starpu_mpi_insert_task"     )
        LIST(APPEND STARPU_name_include  "starpu_mpi.h"               )
    ELSE()
        SET(STARPU_name_library          "starpu-1.0"                 )
        SET(STARPU_name_pkgconfig        "starpu-1.0"                 )
        LIST(APPEND STARPU_name_fct_test "starpu_insert_task"         )
    ENDIF()

    IF(HAVE_CUDA)
        LIST(APPEND STARPU_name_include  "starpu_cuda.h"              )
        LIST(APPEND STARPU_name_include  "starpu_scheduler.h"         )
    ENDIF()

    IF(HAVE_FXT)
        LIST(APPEND STARPU_name_fct_test "starpu_fxt_generate_trace"  )
    ENDIF()

ENDMACRO(STARPU_INFO_FIND)

##
## @end file infoSTARPU.cmake
##
