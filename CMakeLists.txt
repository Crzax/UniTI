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
  set(CMAKE_CXX_FLAGS "-std=c++11 -O2 -march=native ${CMAKE_CXX_FLAGS}")
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
# Find OpenBLAS for high-performance matrix multiplication (optional)
find_path(CBLAS_INCLUDE_DIR cblas.h
  HINTS /usr/include/x86_64-linux-gnu /usr/include /usr/local/include
        /usr/include/openblas
)
find_library(OPENBLAS_LIB openblas
  HINTS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib
        /usr/lib64
)
message(STATUS "OpenBLAS include: ${CBLAS_INCLUDE_DIR}")
message(STATUS "OpenBLAS lib: ${OPENBLAS_LIB}")

add_library(ndarray_backend_cpu MODULE src/ndarray_backend_cpu.cc)

if(CBLAS_INCLUDE_DIR AND OPENBLAS_LIB)
  message(STATUS "OpenBLAS found — using BLAS-accelerated matmul")
  target_compile_definitions(ndarray_backend_cpu PRIVATE USE_OPENBLAS=1)
  target_include_directories(ndarray_backend_cpu PRIVATE ${CBLAS_INCLUDE_DIR})
  target_link_libraries(ndarray_backend_cpu PUBLIC ${LINKER_LIBS} ${OPENBLAS_LIB})
else()
  message(WARNING "OpenBLAS NOT found — CPU matmul will use naive fallback (slow). "
                  "Install OpenBLAS for better performance: apt install libopenblas-dev / yum install openblas-devel")
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

