cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(compressor LANGUAGES CXX C)
find_library(ZFP_LIBRARY NAMES zfp)
find_library(SZ_LIBRARY NAMES SZ)
find_library(MGARD_LIBRARY NAMES mgard)
find_library(ADIOS2_LIBRARY NAMES adios2)
find_library(ZC_LIBRARY NAMES zc)
find_library(GF_LIBRARY NAMES gf)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(Kokkos REQUIRED)
find_path(ADIOS2_INCLUDE_DIR NAMES adios2.h)
find_path(MGARD_INCLUDE_DIR NAMES mgard_nuni.h)
find_path(FTK_INCLUDE_DIR NAMES ftk/utils/zcolor.h)
find_path(GF_INCLUDE_DIR NAMES gaussian_filter.h)
find_library(TIMING_CHRONO_LIBRARY NAMES timing_chrono)
find_path(TIMING_CHRONO_INCLUDE_DIR NAMES timing.h)
# message(${FTK_INCLUDE_DIR})

# compress
add_executable(compress compress.cpp sz_local.cpp zfp_local.cpp mgard_local.cpp 
  mgard_local.h  sz_local.h  zfp_local.h ftk_3D_interface.h)
target_link_libraries(compress MPI::MPI_CXX ${ADIOS2_LIBRARY} ${ZFP_LIBRARY} 
			       ${SZ_LIBRARY} ${MGARD_LIBRARY} ${TIMING_CHRONO_LIBRARY})
target_include_directories(compress PRIVATE ${MGARD_INCLUDE_DIR} PUBLIC ${ADIOS2_INCLUDE_DIR} ${TIMING_CHRONO_INCLUDE_DIR})
set_property(TARGET compress PROPERTY CXX_STANDARD 11)


# zchecker
add_executable(zchecker zchecker.cpp)
target_link_libraries(zchecker ${ZC_LIBRARY} ${ADIOS2_LIBRARY} MPI::MPI_CXX ${TIMING_CHRONO_LIBRARY})
target_include_directories(zchecker PUBLIC ${ADIOS2_INCLUDE_DIR} ${TIMING_CHRONO_INCLUDE_DIR})


# ftk
add_executable(ftk ftk.cpp ftk_3D.cpp ftk_3D_interface.h)
target_link_libraries(ftk MPI::MPI_CXX ${ADIOS2_LIBRARY} ${TIMING_CHRONO_LIBRARY} ${GF_LIBRARY} Kokkos::kokkos)
target_compile_features(ftk PUBLIC cxx_std_17)
target_include_directories(ftk PRIVATE ${FTK_INCLUDE_DIR} PUBLIC ${ADIOS2_INCLUDE_DIR} ${TIMING_CHRONO_INCLUDE_DIR} ${GF_INCLUDE_DIR})
set_property(TARGET ftk PROPERTY CXX_STANDARD 17)

install(TARGETS compress zchecker ftk)
