###
#
# @file      : CMakeLists.txt
#
# @description   : Project MAGMA
#
# @version       :
# @created by    : Cedric Castagnede
# @creation date : 23-01-2012
# @last modified : mar. 17 avril 2012 15:53:16 CEST
#
###

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
INCLUDE(RulesPrecisions)

# Check for the subdirectories 
# ----------------------------
ADD_SUBDIRECTORY(matgen)
ADD_SUBDIRECTORY(lin)

# Generate the magma testing sources for all possible precisions
# --------------------------------------------------------------
SET(MAGMA_TEST_GENERATED "")
SET(ZSRC 
     testing_zgemm.cpp
     testing_zgemv.cpp
     testing_zhemv.cpp
     testing_zsymv.cpp
###
     testing_zpotrf.cpp
     testing_zpotrf_mgpu.cpp
     testing_zpotri.cpp
     testing_zposv.cpp
     testing_zgetrf.cpp
     testing_zgesv.cpp
     testing_zgeqrf.cpp
     testing_zgeqlf.cpp
     testing_zgelqf.cpp
     testing_zgeqp3.cpp
     testing_dgeqp3.cpp
     testing_zungqr.cpp
###
     testing_zpotrf_gpu.cpp
     testing_zpotri_gpu.cpp
     testing_zposv_gpu.cpp
     testing_zgetrf_gpu.cpp
     testing_zgetrf_mgpu.cpp
     testing_zgetri_gpu.cpp
     testing_zgesv_gpu.cpp
     testing_zgeqrf_gpu.cpp
     testing_zgeqrf_mgpu.cpp
     testing_zgelqf_gpu.cpp
     testing_zgeqrs_gpu.cpp
     testing_zgeqrs3_gpu.cpp
     testing_zungqr_gpu.cpp
     testing_zlarfb_gpu.cpp
###
     testing_zgebrd.cpp
     testing_zgehrd.cpp
     testing_zhetrd.cpp
     testing_zgeev.cpp
     testing_dgeev.cpp
     testing_zgesvd.cpp
     testing_zheevd.cpp
     testing_dsyevd.cpp
     testing_zhegvd.cpp
     testing_dsygvd.cpp
     testing_zhegvdx.cpp
     testing_zhebbd.cpp
###
     testing_zhetrd_gpu.cpp
     testing_zheevd_gpu.cpp
###
     testing_zcposv_gpu.cpp
     testing_zcgesv_gpu.cpp
     testing_zcgeqrsv_gpu.cpp
###
     testing_zlacpy.cpp
     testing_auxiliary.cpp
     testing_zblas.cpp
     testing_zher2k_mgpu.cpp
     testing_zhemm_mgpu.cpp
     testing_zsymmetrize.cpp
     )

IF(HAVE_PGI)
    SET(CUDA_INTERFACE ${CUDA_TOOLKIT_ROOT_DIR}/src/fortran_thunking.c)
    SET(ZSRCF testing_zgetrf_gpu_f.cuf)
ELSE()
    SET(CUDA_INTERFACE ${CUDA_TOOLKIT_ROOT_DIR}/src/fortran.c)
    SET(ZSRCF testing_zgetrf_gpu_f.f90
          testing_zgetrf_f.f90
          )
ENDIF()

#SET(ZSRC ${ZSRC}
#         ${ZSRCF}
#         )

precisions_rules_py(MAGMA_TEST_GENERATED
            "${ZSRC}"
            PRECISIONS ${MAGMA_PRECISION})

# Define the list of sources
# --------------------------
#SET(MAGMA_TEST ${CUDA_INTERFACE}
#               ${MAGMA_TEST_GENERATED}
#               )


# Add dependencies on the header files
# ------------------------------------
#STRING(REGEX REPLACE ".c" ".o" CUDA_INTERFACE_OBJ "${CUDA_INTERFACE}")
#SET_SOURCE_FILES_PROPERTIES(${ZSRCF}
# PROPERTIES OBJECTS_DEPENDS ${CUDA_INTERFACE_OBJ}
# )

# Add include and link directories
# --------------------------------
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR}/include)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/testing)
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR}/testing)

#MESSAGE(STATUS "flags: ${PLASMA_LDFLAGS}")

# Add tests (C/CPP)
# -----------------
SET(libs_for_tests 
    magma 
    magmablas 
    magma 
    cublas 
    lapacktest 
    tmglib 
    ${EXTRA_LIBS} 
    ${CMAKE_EXTRA_LDFLAGS_F}
    )

FOREACH(_test_src ${MAGMA_TEST_GENERATED})
    GET_FILENAME_COMPONENT(_name_exe ${_test_src} NAME_WE)
    ADD_EXECUTABLE(${_name_exe} ${_test_src})
    SET_PROPERTY(TARGET ${_name_exe} PROPERTY LINKER_LANGUAGE Fortran)
    TARGET_LINK_LIBRARIES(${_name_exe} ${libs_for_tests})
    ADD_TEST(NAME ${_name_exe} 
             COMMAND ${_name_exe})
ENDFOREACH()

# Add tests (Fortran)
# -----------------
#FOREACH(_test_src ${ZSRCF})
#    GET_FILENAME_COMPONENT(_name_exe ${_test_src} NAME_WE)
#    ADD_EXECUTABLE(${_name_exe} ${_test_src} ${CUDA_INTERFACE})
#    SET_PROPERTY(TARGET ${_name_exe} PROPERTY LINKER_LANGUAGE Fortran)
#    TARGET_LINK_LIBRARIES(${_name_exe} magma magmablas magma cublas lapacktest tmglib ${EXTRA_LIBS})
#    ADD_TEST(NAME ${_name_exe}
#             COMMAND ${_name_exe})
#ENDFOREACH()
    
###
### END CMakeLists.txt
###
