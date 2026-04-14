cmake_minimum_required(VERSION 3.5)
project(uniti C CXX)
if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()

# find correct version of Python
# Use the python3 from PATH (conda / venv / system) so the .so suffix matches.
execute_process(COMMAND python3-config --prefix
  OUTPUT_VARIABLE Python_ROOT_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND which python3
  OUTPUT_VARIABLE _PYTHON3_EXEC
  OUTPUT_STRIP_TRAILING_WHITESPACE)
set(Python_EXECUTABLE ${_PYTHON3_EXEC})
find_package(Python COMPONENTS Development Interpreter REQUIRED)
include_directories(${Python_INCLUDE_DIRS})

# find pybind – prefer the pip/conda-installed version over any system pybind11
execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
  RESULT_VARIABLE __pybind_exit_code
  OUTPUT_VARIABLE __pybind_path
  OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 PATHS ${__pybind_path} NO_DEFAULT_PATH)


if(NOT MSVC)
  # On Apple Silicon (arm64), -march=native may not be recognized by all compilers;
  # use -mcpu=apple-m1 (covers M1–M5) for optimal NEON/AMX code generation.
  if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "-std=c++11 -O2 -march=native ${CMAKE_CXX_FLAGS}")
  endif()
  set(CMAKE_CUDA_STANDARD 14)
else()
  set(CMAKE_CXX_FLAGS "/std:c++11 -O2 -march=native ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 14)
endif()

include_directories(SYSTEM ${pybind11_INCLUDE_DIRS})
list(APPEND LINKER_LIBS ${pybind11_LIBRARIES})


###################
### CPU BACKEND ###
###################
# BLAS detection: prefer Apple Accelerate on macOS, OpenBLAS on Linux.
# Apple Accelerate provides highly optimized BLAS/LAPACK for Apple Silicon (M1–M5),
# using the AMX coprocessor for matrix operations — no extra installation needed.

set(BLAS_FOUND FALSE)

if(APPLE)
  # macOS: use Apple Accelerate framework (built-in, zero-install)
  # Accelerate includes a full cblas.h and vecLib with AMX acceleration on Apple Silicon.
  find_library(ACCELERATE_FRAMEWORK Accelerate)
  if(ACCELERATE_FRAMEWORK)
    set(BLAS_FOUND TRUE)
    set(BLAS_BACKEND "Apple Accelerate")
    message(STATUS "Apple Accelerate found — using AMX-accelerated BLAS for Apple Silicon")
  endif()
endif()

if(NOT BLAS_FOUND)
  # Linux / other: find OpenBLAS
  find_path(CBLAS_INCLUDE_DIR cblas.h
    HINTS /usr/include/x86_64-linux-gnu /usr/include /usr/local/include
          /usr/include/openblas
  )
  find_library(OPENBLAS_LIB_MT openblasp
    HINTS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib /usr/lib64
  )
  find_library(OPENBLAS_LIB_ST openblas
    HINTS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib /usr/lib64
  )
  if(OPENBLAS_LIB_MT)
    set(OPENBLAS_LIB ${OPENBLAS_LIB_MT})
    message(STATUS "Using multi-threaded OpenBLAS: ${OPENBLAS_LIB}")
  elseif(OPENBLAS_LIB_ST)
    set(OPENBLAS_LIB ${OPENBLAS_LIB_ST})
    message(STATUS "Using single-threaded OpenBLAS: ${OPENBLAS_LIB}")
  endif()
  if(CBLAS_INCLUDE_DIR AND OPENBLAS_LIB)
    set(BLAS_FOUND TRUE)
    set(BLAS_BACKEND "OpenBLAS")
  endif()
endif()

add_library(ndarray_backend_cpu MODULE src/ndarray_backend_cpu.cc)

if(BLAS_FOUND)
  target_compile_definitions(ndarray_backend_cpu PRIVATE USE_OPENBLAS=1)
  if(APPLE AND ACCELERATE_FRAMEWORK)
    message(STATUS "BLAS backend: Apple Accelerate (AMX-accelerated on Apple Silicon)")
    target_link_libraries(ndarray_backend_cpu PUBLIC ${LINKER_LIBS} ${ACCELERATE_FRAMEWORK})
  else()
    message(STATUS "BLAS backend: OpenBLAS")
    target_include_directories(ndarray_backend_cpu PRIVATE ${CBLAS_INCLUDE_DIR})
    target_link_libraries(ndarray_backend_cpu PUBLIC ${LINKER_LIBS} ${OPENBLAS_LIB})
  endif()
else()
  message(WARNING "No BLAS library found — CPU matmul will use naive fallback (slow). "
                  "On macOS: Accelerate should be available by default. "
                  "On Linux: apt install libopenblas-dev / yum install openblas-devel")
  target_link_libraries(ndarray_backend_cpu PUBLIC ${LINKER_LIBS})
endif()

pybind11_extension(ndarray_backend_cpu)
pybind11_strip(ndarray_backend_cpu)


# directly output to ffi folder
set_target_properties(ndarray_backend_cpu
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/uniti/backend_ndarray
  CXX_VISIBILITY_PRESET "hidden"
)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set_property(TARGET ndarray_backend_cpu PROPERTY LINK_OPTIONS -undefined dynamic_lookup)
endif()



####################
### CUDA BACKEND ###
####################
find_package(CUDA)
if(CUDA_FOUND)
  message(STATUS "Found cuda, building cuda backend")

  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
  list(APPEND LINKER_LIBS ${CUDA_CUDART_LIBRARY})

  # invoke nvidia smi to detect if we really have a GPU
  execute_process(COMMAND "nvidia-smi" ERROR_QUIET  RESULT_VARIABLE NV_RET)
  if(NV_RET EQUAL "0")
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
  else()
    # set to 3.7 the flag of K80
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 3.7)
  endif()

  # set arch flags properly
  CUDA_ADD_LIBRARY(ndarray_backend_cuda MODULE src/ndarray_backend_cuda.cu OPTIONS ${ARCH_FLAGS})

  target_link_libraries(ndarray_backend_cuda ${LINKER_LIBS})
  pybind11_extension(ndarray_backend_cuda)
  pybind11_strip(ndarray_backend_cuda)

  # directly output to ffi folder
  set_target_properties(ndarray_backend_cuda
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/uniti/backend_ndarray
    CXX_VISIBILITY_PRESET "hidden"
    CUDA_VISIBILITY_PRESET "hidden"
)

endif()


#####################
### METAL BACKEND ###
#####################
# Apple Metal GPU backend — only built on macOS with Metal framework
if(APPLE)
  find_library(METAL_FRAMEWORK Metal)
  find_library(FOUNDATION_FRAMEWORK Foundation)
  if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
    message(STATUS "Found Metal framework — building Metal GPU backend")

    add_library(ndarray_backend_metal MODULE src/ndarray_backend_metal.mm)
    target_link_libraries(ndarray_backend_metal PUBLIC ${LINKER_LIBS} ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
    target_compile_options(ndarray_backend_metal PRIVATE -fobjc-arc)

    pybind11_extension(ndarray_backend_metal)
    pybind11_strip(ndarray_backend_metal)

    set_target_properties(ndarray_backend_metal
      PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/uniti/backend_ndarray
      CXX_VISIBILITY_PRESET "hidden"
    )
    set_property(TARGET ndarray_backend_metal PROPERTY LINK_OPTIONS -undefined dynamic_lookup)
  endif()
endif()

