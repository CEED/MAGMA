###
#
# @file          : infoSTARPU.cmake
#
# @description   :
#
# @version       :
# @created by    : Cedric Castagnede
# @creation date : 04-04-2012
# @last modified : Thu 05 Jul 2012 08:28:50 PM CEST
#
###

MACRO(STARPU_INFO_INSTALL)
    # Define web link of starpu
    # -------------------------
    IF(NOT DEFINED STARPU_URL)
        SET(STARPU_URL     "http://runtime.bordeaux.inria.fr/StarPU/files/starpu-1.0.1.tar.gz")
    ENDIF()

    # Define tarball of starpu
    # ------------------------
    IF(NOT DEFINED STARPU_TARBALL)
        SET(STARPU_TARBALL "starpu-1.0.1.tar.gz")
    ENDIF()

    # Define md5sum of eigen
    # ----------------------
    IF(DEFINED STARPU_URL OR DEFINED STARPU_TARBALL)
        SET(STARPU_MD5SUM  "72e9d92057f2b88483c27aca78c53316")
    ENDIF()

    # Define repository of starpu
    # ---------------------------
    IF(NOT DEFINED STARPU_SVN_REP)
        SET(STARPU_REPO_MODE "SVN"                                   )
        SET(STARPU_SVN_REP   "https://scm.gforge.inria.fr/svn/starpu")
        SET(STARPU_SVN_ID    "anonsvn"                               )
        SET(STARPU_SVN_PWD   "anonsvn"                               )
    ENDIF()

   # Define dependencies
   # -------------------
   SET(STARPU_DEPENDENCIES "hwloc;fxt;mpi")

ENDMACRO(STARPU_INFO_INSTALL)

MACRO(STARPU_INFO_FIND)
    # Define parameters for FIND_MY_PACKAGE
    # -------------------------------------
    SET(STARPU_type_library             "C"                          )
    IF(MORSE_USE_MPI)
        SET(STARPU_name_library         "starpumpi-1.0;starpu-1.0"   )
        SET(STARPU_name_pkgconfig       "starpumpi-1.0"              )
        SET(STARPU_name_fct_test        "starpu_mpi_insert_task"     )
    ELSE(MORSE_USE_MPI)
        SET(STARPU_name_library         "starpu-1.0"                 )
        SET(STARPU_name_pkgconfig       "starpu-1.0"                 )
        SET(STARPU_name_fct_test        "starpu_insert_task"         )
    ENDIF(MORSE_USE_MPI)
    SET(STARPU_name_include             "starpu.h;starpu_profiling.h")
    IF(MORSE_USE_MPI)
        LIST(APPEND STARPU_name_include "starpu_mpi.h"               )
    ENDIF(MORSE_USE_MPI)
    IF(MORSE_USE_CUDA)
        LIST(APPEND STARPU_name_include "starpu_cuda.h"              )
        LIST(APPEND STARPU_name_include "starpu_scheduler.h"         )
    ENDIF(MORSE_USE_CUDA)
    SET(STARPU_name_include_suffix      "starpu/1.0"                 )
    SET(STARPU_name_binary              "STARPU_name_binary-NOTFOUND")
    #SET(STARPU_name_binary             "starpu_calibrate_bus"       )

ENDMACRO(STARPU_INFO_FIND)

###
### END infoSTARPU.cmake
###
