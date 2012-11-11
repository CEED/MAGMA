###
#
#  @file installOPENMPI.cmake
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

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
INCLUDE(installExternalPACKAGE)

MACRO(INSTALL_OPENMPI _MODE)

    # Define prefix paths
    # -------------------
    IF(MORSE_SEPARATE_PROJECTS)
        SET(MPI_PATH ${CMAKE_INSTALL_PREFIX}/mpi)
    ELSE(MORSE_SEPARATE_PROJECTS)
        SET(MPI_PATH ${CMAKE_INSTALL_PREFIX})
    ENDIF(MORSE_SEPARATE_PROJECTS)

    # Define options
    # --------------
    UNSET(MPI_CONFIG_OPTS)
    LIST(APPEND MPI_CONFIG_OPTS --prefix=${MPI_PATH})
    LIST(APPEND MPI_CONFIG_OPTS --enable-mpi-thread-multiple)
    LIST(APPEND MPI_CONFIG_OPTS --enable-shared)
    IF(NOT BUILD_SHARED_LIBS)
        LIST(APPEND MPI_CONFIG_OPTS --enable-static)
    ENDIF()

    # Define steps of installation
    # ----------------------------
    SET(MPI_SOURCE_PATH ${CMAKE_BINARY_DIR}/externals/mpi)
    SET(MPI_BUILD_PATH  ${CMAKE_BINARY_DIR}/externals/mpi)
    SET(MPI_CONFIG_CMD  ./configure)
    SET(MPI_BUILD_CMD   ${CMAKE_MAKE_PROGRAM})
    SET(MPI_INSTALL_CMD ${CMAKE_MAKE_PROGRAM} install)

    # Install the external package
    # ----------------------------
    INSTALL_EXTERNAL_PACKAGE("mpi" "${MPI_USED_MODE}")

    # Set linker flags
    # ----------------
    UNSET(MPI_C_LIBRARIES CACHE)
    UNSET(MPI_CXX_LIBRARIES CACHE)
    UNSET(MPI_Fortran_LIBRARIES CACHE)

    UNSET(MPI_LDFLAGS)
    UNSET(MPI_LIBRARY)
    UNSET(MPI_LIBRARIES)
    UNSET(MPI_LIBRARY_PATH)
    UNSET(OPENMPI_EXTRA_LIBRARY)
    UNSET(MPI_LIBRARY CACHE)
    UNSET(MPI_LIBRARY_PATH CACHE)

    #
    # There is a problem here: after make step in openmpi
    #                          all executable are not available
    #                          There are wrapper of orterun and opal_wrapper,
    #                          and they work only after make install step
    #
    # Set value according to FindMPI (mpiexec)
    #SET(MPIEXEC               "${MPI_BUILD_PATH}/orte/tools/orterun/.libs/orterun")
    #SET(MPIEXEC_MAX_NUMPROCS  "2"                     )
    #SET(MPIEXEC_NUMPROC_FLAGS "-np"                   )
    #SET(MPI_BINARY_PATH      "${MPI_PATH}/bin"        )
    #SET(MPI_Fortran_COMPILER "${MPI_PATH}/bin/mpif90" )
    #SET(MPI_C_COMPILER       "${MPI_PATH}/bin/mpicc"  )
    #SET(MPI_CXX_COMPILER     "${MPI_PATH}/bin/mpicixx")

    # Set value according to FindMPI (library C)
    SET(MPI_C_LINK_FLAGS        "-Wl,--export-dynamic"                   )
    SET(MPI_C_INCLUDE_PATH      "${MPI_BUILD_PATH}/ompi/include"         )
    LIST(APPEND MPI_C_LIBRARIES
         "${MPI_BUILD_PATH}/ompi/.libs/libmpi.${MORSE_LIBRARY_EXTENSION}")

    # Set value according to FindMPI (libraries CXX)
    SET(MPI_CXX_LINK_FLAGS        "-Wl,--export-dynamic"                             )
    SET(MPI_CXX_INCLUDE_PATH      "${MPI_BUILD_PATH}/ompi/include"                   )
    LIST(APPEND MPI_CXX_LIBRARIES
         "${MPI_BUILD_PATH}/ompi/mpi/cxx/.libs/libmpi_cxx.${MORSE_LIBRARY_EXTENSION}")
    LIST(APPEND MPI_CXX_LIBRARIES
         "${MPI_BUILD_PATH}/ompi/.libs/libmpi.${MORSE_LIBRARY_EXTENSION}"            )

    # Set value according to FindMPI (libraries F90)
    SET(MPI_Fortran_LINK_FLAGS        "-Wl,--export-dynamic"                         )
    SET(MPI_Fortran_INCLUDE_PATH      "${MPI_BUILD_PATH}/ompi/include"               )
    LIST(APPEND MPI_Fortran_LIBRARIES
         "${MPI_BUILD_PATH}/ompi/mpi/f90/.libs/libmpi_f90.${MORSE_LIBRARY_EXTENSION}")
    LIST(APPEND MPI_Fortran_LIBRARIES
         "${MPI_BUILD_PATH}/ompi/mpi/f77/.libs/libmpi_f77.${MORSE_LIBRARY_EXTENSION}")
    LIST(APPEND MPI_Fortran_LIBRARIES
         "${MPI_BUILD_PATH}/ompi/.libs/libmpi.${MORSE_LIBRARY_EXTENSION}"            )

    # Set value according to FindMPI (libraries OPAL)
    LIST(APPEND OPENMPI_EXTRA_LIBRARY
         "${MPI_BUILD_PATH}/orte/.libs/libopen-rte.${MORSE_LIBRARY_EXTENSION}")
    LIST(APPEND OPENMPI_EXTRA_LIBRARY
         "${MPI_BUILD_PATH}/opal/.libs/libopen-pal.${MORSE_LIBRARY_EXTENSION}")

    # Set value for MORSE buildsystem
    SET(MPI_INCLUDE_PATH         "${MPI_BUILD_PATH}/ompi/include"      )
    LIST(APPEND MPI_LIBRARY_PATH "${MPI_BUILD_PATH}/ompi/.libs"        )
    LIST(APPEND MPI_LIBRARY_PATH "${MPI_BUILD_PATH}/ompi/mpi/cxx/.libs")
    LIST(APPEND MPI_LIBRARY_PATH "${MPI_BUILD_PATH}/ompi/mpi/f90/.libs")
    LIST(APPEND MPI_LIBRARY_PATH "${MPI_BUILD_PATH}/ompi/mpi/f77/.libs")
    LIST(APPEND MPI_LIBRARY_PATH "${MPI_BUILD_PATH}/orte/.libs"        )
    LIST(APPEND MPI_LIBRARY_PATH "${MPI_BUILD_PATH}/opal/.libs"        )
    LIST(APPEND MPI_LIBRARY      "${MPI_C_LIBRARIES}"                  )
    LIST(APPEND MPI_LIBRARY      "${MPI_CXX_LIBRARIES}"                )
    LIST(APPEND MPI_LIBRARY      "${MPI_Fortran_LIBRARIES}"            )
    LIST(APPEND MPI_LIBRARY      "${OPENMPI_EXTRA_LIBRARY}"            )
    LIST(APPEND MPI_LIBRARIES    "mpi_cxx"                             ) 
    LIST(APPEND MPI_LIBRARIES    "mpi_f90"                             )
    LIST(APPEND MPI_LIBRARIES    "mpi_f77"                             )
    LIST(APPEND MPI_LIBRARIES    "mpi"                                 )
    LIST(APPEND MPI_LIBRARIES    "open-rte"                            )
    LIST(APPEND MPI_LIBRARIES    "open-pal"                            )
    FOREACH(_path ${MPI_LIBRARY_PATH})
        SET(MPI_LDFLAGS          "${MPI_LDFLAGS} -L${_path}"           )
    ENDFOREACH()
    FOREACH(_lib ${MPI_LIBRARIES})
        SET(MPI_LDFLAGS          "${MPI_LDFLAGS} -l${_lib}"            )
    ENDFOREACH()

ENDMACRO(INSTALL_OPENMPI)

##
## @end file installOPENMPI.cmake
##

