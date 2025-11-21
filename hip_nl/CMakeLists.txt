cmake_minimum_required(VERSION 3.21)
project(hip_nl LANGUAGES CXX HIP)

# Set ROCm path
if(NOT DEFINED ROCM_PATH)
    set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm installation")
endif()

# Add ROCm cmake modules to path
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
list(APPEND CMAKE_MODULE_PATH ${ROCM_PATH}/lib/cmake/hip)

# Find HIP
find_package(hip REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# HIP source files
set(HIP_SOURCE_FILES
    hip_neighborlist_simple.hip
)

# Mark .hip files as HIP source
set_source_files_properties(${HIP_SOURCE_FILES} PROPERTIES LANGUAGE HIP)

# Create shared library
add_library(hip_nl SHARED ${HIP_SOURCE_FILES})

# Link HIP libraries
target_link_libraries(hip_nl PRIVATE hip::device)

# Set output directory
set_target_properties(hip_nl PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    PREFIX "lib"
    HIP_ARCHITECTURES "gfx1100;gfx1102"
)

# Compiler flags for optimization
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(hip_nl PRIVATE -O3 -ffast-math)
endif()

# Install target
install(TARGETS hip_nl
    LIBRARY DESTINATION lib
)
