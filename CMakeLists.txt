cmake_minimum_required(VERSION 2.8.12)

# Package name (without the "rtmaps_" prefix)
set(PCK cuda_component_sample)
project("${PCK}.u")

if (NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD          14)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS        OFF)
endif()

if (NOT DEFINED RTMAPS_SDKDIR)
    if (DEFINED ENV{RTMAPS_SDKDIR})
        set(RTMAPS_SDKDIR "$ENV{RTMAPS_SDKDIR}" CACHE PATH "")
    else()
        message(FATAL_ERROR "RTMAPS_SDKDIR not defined")
    endif()
endif()

include("${RTMAPS_SDKDIR}/templates.u/rtmaps.cmake")

# FindCUDA https://cmake.org/cmake/help/v3.5/module/FindCUDA.html
#
# You may or may not need to export the following before running RTMaps
# export LD_LIBRARY_PATH="/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
# 7.1 Mandatory actions: https://developer.download.nvidia.com/compute/cuda/10.0/Prod/docs/sidebar/CUDA_Installation_Guide_Linux.pdf
if (NOT DEFINED ENV{CUDA_BIN_PATH})
    set(ENV{CUDA_BIN_PATH} "/usr/local/cuda-10.0/bin")
endif()
find_package(CUDA 10 REQUIRED)
message(STATUS "CUDA Version: ${CUDA_VERSION} @ ${CUDA_TOOLKIT_ROOT_DIR}")

# good compiler warnings
if (MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-fdiagnostics-show-option -Wall -Wextra -pedantic -Werror)
endif()

if (WIN32)
	enable_language(CUDA)
endif()

file(GLOB_RECURSE sources
     "src/*.cu"
     "src/*.h"   "src/*.c"
     "src/*.hpp" "src/*.cpp"
     "src/*.hxx" "src/*.cxx"
)

# Create the pck target
cuda_add_library(${PCK} SHARED ${sources})
configure_target_as_rtmaps_package(${PCK} PCKINFO "rtmaps_package.pckinfo")

target_include_directories(${PCK} PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(${PCK} cuda ${CUDA_LIBRARIES})
target_compile_definitions(${PCK} PRIVATE MAPS_COMPILING_PCK)
