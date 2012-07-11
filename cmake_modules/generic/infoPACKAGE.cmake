###
#
# @file          : infoPACKAGE.cmake
#
# @description   :
#
# @version       :
# @created by    : Cedric Castagnede
# @creation date : 09-03-2012
# @last modified : mar. 12 juin 2012 16:43:45 CEST
#
###

# Macro to define test functions 
# ------------------------------
MACRO(INFO_INSTALL_PACKAGE)

    INCLUDE(infoBLAS)
    BLAS_INFO_INSTALL()
    INCLUDE(infoLAPACK)
    LAPACK_INFO_INSTALL()
    INCLUDE(infoCBLAS)
    CBLAS_INFO_INSTALL()
    INCLUDE(infoLAPACKE)
    LAPACKE_INFO_INSTALL()
    INCLUDE(infoPLASMA)
    PLASMA_INFO_INSTALL()
    INCLUDE(infoOPENMPI)
    MPI_INFO_INSTALL()
    INCLUDE(infoFXT)
    FXT_INFO_INSTALL()
    INCLUDE(infoHWLOC)
    HWLOC_INFO_INSTALL()
    INCLUDE(infoSTARPU)
    STARPU_INFO_INSTALL()

ENDMACRO(INFO_INSTALL_PACKAGE)

MACRO(INFO_FIND_PACKAGE)

    INCLUDE(infoBLAS)
    BLAS_INFO_FIND()
    INCLUDE(infoLAPACK)
    LAPACK_INFO_FIND()
    INCLUDE(infoCBLAS)
    CBLAS_INFO_FIND()
    INCLUDE(infoLAPACKE)
    LAPACKE_INFO_FIND()
    INCLUDE(infoPLASMA)
    PLASMA_INFO_FIND()
    INCLUDE(infoOPENMPI)
    MPI_INFO_FIND()
    INCLUDE(infoFXT)
    FXT_INFO_FIND()
    INCLUDE(infoHWLOC)
    HWLOC_INFO_FIND()
    INCLUDE(infoSTARPU)
    STARPU_INFO_FIND()

ENDMACRO(INFO_FIND_PACKAGE)

###
### END infoPACKAGE.cmake
###
