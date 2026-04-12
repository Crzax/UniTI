#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace uniti {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides
__device__ size_t GetOriIndex(size_t index, CudaVec shape,
                              CudaVec strides, size_t offset) {
  size_t idxs[MAX_VEC_SIZE];
  size_t ndim = shape.size;
  size_t cur_size = 1, pre_size = 1;
  for (int i = ndim - 1; i >= 0; --i) {
    cur_size *= shape.data[i];
    idxs[i] = (index % cur_size) / pre_size;
    pre_size = cur_size;
  }
  size_t ori_off = offset;
  for (size_t i = 0; i < ndim; ++i) {
    ori_off += idxs[i] * strides.data[i];
  }
  return ori_off;
}


__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  if (gid < size) {
    out[gid] = a[GetOriIndex(gid, shape, strides, offset)];
  }
  /// END SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}

__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[GetOriIndex(gid, shape, strides, offset)] = a[gid];
  }
}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {

  CudaDims dim = CudaOneDim(a.size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}

__global__ void ScalarSetitemKernel(scalar_t val, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[GetOriIndex(gid, shape, strides, offset)] = val;
  }
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {

  CudaDims dim = CudaOneDim(out->size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////


__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA arrays.
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a + b'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add a scalar value to every element of a CUDA array.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a + val'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Launch the ScalarAddKernel that will add the scalar 'val' to each element of array 'a', 
  // and store the result in array 'out'.
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */
__global__ void EwiseMulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[gid] = a[gid] * b[gid];
  }
}

void EwiseMul(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseMulKernel<<<dim.grid,dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarMulKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[gid] = a[gid] * val;
  }
}

void ScalarMul(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarMulKernel<<<dim.grid,dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseDivKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[gid] = a[gid] / b[gid];
  }
}

void EwiseDiv(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseDivKernel<<<dim.grid,dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarDivKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[gid] = a[gid] / val;
  }
}

void ScalarDiv(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarDivKernel<<<dim.grid,dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void ScalarPowerKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[gid] = pow(a[gid], val);
  }
}

void ScalarPower(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarPowerKernel<<<dim.grid,dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseMaximumKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[gid] = max(a[gid], b[gid]);
  }
}

void EwiseMaximum(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseMaximumKernel<<<dim.grid,dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarMaximumKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[gid] = max(a[gid], val);
  }
}

void ScalarMaximum(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarMaximumKernel<<<dim.grid,dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseEqKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[gid] = a[gid] == b[gid] ? 1.0 : 0.0;
  }
}

void EwiseEq(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseEqKernel<<<dim.grid,dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarEqKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[gid] = a[gid] == val ? 1.0 : 0.0;
  }
}

void ScalarEq(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarEqKernel<<<dim.grid,dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseGeKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[gid] = a[gid] >= b[gid] ? 1.0 : 0.0;
  }
}

void EwiseGe(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseGeKernel<<<dim.grid,dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarGeKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[gid] = a[gid] >= val ? 1.0 : 0.0;
  }
}

void ScalarGe(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarGeKernel<<<dim.grid,dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseLogKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[gid] = log(a[gid]);
  }
}

void EwiseLog(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseLogKernel<<<dim.grid,dim.block>>>(a.ptr, out->ptr, out->size);
}

__global__ void EwiseExpKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[gid] = exp(a[gid]);
  }
}

void EwiseExp(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseExpKernel<<<dim.grid,dim.block>>>(a.ptr, out->ptr, out->size);
}

__global__ void EwiseTanhKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[gid] = tanh(a[gid]);
  }
}

void EwiseTanh(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseTanhKernel<<<dim.grid,dim.block>>>(a.ptr, out->ptr, out->size);
}

__global__ void EwiseSinKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[gid] = sinf(a[gid]);
  }
}

void EwiseSin(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseSinKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

__global__ void EwiseCosKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[gid] = cosf(a[gid]);
  }
}

void EwiseCos(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseCosKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

__global__ void ArangeKernel(scalar_t* out, size_t n) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < n) {
    out[gid] = static_cast<scalar_t>(gid);
  }
}

void Arange(CudaArray* out, size_t n) {
  CudaDims dim = CudaOneDim(n);
  ArangeKernel<<<dim.grid, dim.block>>>(out->ptr, n);
}

__global__ void TriuMaskKernel(scalar_t* out, size_t rows, size_t cols, int k, scalar_t mask_val) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = rows * cols;
  if (gid < total) {
    size_t i = gid / cols;
    size_t j = gid % cols;
    out[gid] = (static_cast<int>(j) >= static_cast<int>(i) + k) ? mask_val : 0.0f;
  }
}

void TriuMask(CudaArray* out, size_t rows, size_t cols, int k, scalar_t mask_val) {
  size_t total = rows * cols;
  CudaDims dim = CudaOneDim(total);
  TriuMaskKernel<<<dim.grid, dim.block>>>(out->ptr, rows, cols, k, mask_val);
}

__global__ void EmbeddingLookupKernel(const scalar_t* weight, const scalar_t* ids,
                                       scalar_t* out, size_t num_ids, size_t embedding_dim) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = num_ids * embedding_dim;
  if (gid < total) {
    size_t token_idx = gid / embedding_dim;
    size_t dim_idx = gid % embedding_dim;
    int vocab_idx = static_cast<int>(ids[token_idx]);
    out[gid] = weight[vocab_idx * embedding_dim + dim_idx];
  }
}

void EmbeddingLookup(const CudaArray& weight, const CudaArray& ids, CudaArray* out,
                     size_t num_ids, size_t embedding_dim) {
  size_t total = num_ids * embedding_dim;
  CudaDims dim = CudaOneDim(total);
  EmbeddingLookupKernel<<<dim.grid, dim.block>>>(weight.ptr, ids.ptr, out->ptr, num_ids, embedding_dim);
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

// 无优化
// __global__ void MatmulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, uint32_t M, uint32_t N,
//             uint32_t P) {
//   size_t i = blockIdx.x * blockDim.x + threadIdx.x;
//   size_t j = blockIdx.y * blockDim.y + threadIdx.y;
//   if (i < M && j < P) {
//     scalar_t tmp = 0;
//     for (size_t k = 0; k < N; ++k)
//       tmp += a[i * N + k] * b[k * P + j];
//     out[i * P + j] = tmp;
//   }
// }

// void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
//             uint32_t P) {
//   /**
//    * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
//    * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
//    * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
//    * over (i,j) entries in the output array.  However, to really get the full benefit of this
//    * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
//    * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
//    * the CPU backend, here you should implement a single function that works across all size
//    * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
//    * implementations, this function here will largely just set up the kernel call, and you should
//    * implement the logic in a separate MatmulKernel() call.
//    * 
//    *
//    * Args:
//    *   a: compact 2D array of size m x n
//    *   b: comapct 2D array of size n x p
//    *   out: compact 2D array of size m x p to write the output to
//    *   M: rows of a / out
//    *   N: columns of a / rows of b
//    *   P: columns of b / out
//    */

//   /// BEGIN SOLUTION
//   uint32_t TILE_DIM = 32; 
//   dim3 block(TILE_DIM, TILE_DIM, 1);
//   dim3 grid((M + TILE_DIM - 1) / TILE_DIM, (P + TILE_DIM - 1) / TILE_DIM, 1);
//   MatmulKernel<<<grid, block>>>(a.ptr,b.ptr,out->ptr,M,N,P);
//   /// END SOLUTION
// }

// 没有考虑REGISTER TILE
// #define TILE_DIM 32
// __global__ void MatmulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, 
//                              uint32_t M, uint32_t N, uint32_t P) {
//   size_t tx = threadIdx.y, ty = threadIdx.x;
//   size_t row = blockIdx.y * blockDim.y + tx, col = blockIdx.x * blockDim.x + ty;
//   __shared__ scalar_t sA[TILE_DIM][TILE_DIM];
//   __shared__ scalar_t sB[TILE_DIM][TILE_DIM];
//   scalar_t c = 0;
//   for (size_t i = 0; i < (N + TILE_DIM - 1) / TILE_DIM; ++i) {
//     if (row < M && (i * TILE_DIM + ty < N)) {
//       sA[tx][ty] = a[row * N + i * TILE_DIM + ty];
//     } else {
//       sA[tx][ty] = 0;
//     }
//     if (col < P && (i * TILE_DIM + tx < N)) {
//       sB[tx][ty] = b[(i * TILE_DIM + tx) * P + col];
//     } else {
//       sB[tx][ty] = 0;
//     }
//     __syncthreads();
//     for (size_t j = 0; j < TILE_DIM; ++j) {
//       c += sA[tx][j] * sB[j][ty];
//     }
//     __syncthreads();
//   }
//   if (row < M && col < P) out[row * P + col] = c;
// }

// void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
//             uint32_t P) {
//   dim3 block(TILE_DIM, TILE_DIM, 1);
//   dim3 grid((P + TILE_DIM - 1)/TILE_DIM, (M + TILE_DIM - 1)/TILE_DIM, 1);
//   MatmulKernel<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
// }

/*
 * High-performance SGEMM kernel.
 *
 * Each thread-block computes a BM x BN tile of C.
 * Shared memory tiles: A -> BM x BK, B -> BK x BN.
 * Each thread computes a TM x TN sub-tile via register blocking.
 *
 * Block config: 256 threads (16 x 16), each handles 8x8 output elements.
 * Covers 128 x 128 output per block, iterating over K in steps of 8.
 */
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

__global__ void MatmulKernel(const scalar_t* __restrict__ A,
                              const scalar_t* __restrict__ B,
                              scalar_t* __restrict__ C,
                              uint32_t M, uint32_t N, uint32_t P) {
  const int bx = blockIdx.x;  // column block index
  const int by = blockIdx.y;  // row block index
  const int tx = threadIdx.x; // 0..15
  const int ty = threadIdx.y; // 0..15
  const int tid = ty * blockDim.x + tx; // 0..255

  // Each thread computes TM x TN = 8x8 outputs.
  // Thread (ty, tx) covers rows [by*BM + ty*TM .. +TM) and cols [bx*BN + tx*TN .. +TN).
  const int row0 = by * BM + ty * TM;
  const int col0 = bx * BN + tx * TN;

  // Shared memory for A-tile (BM x BK) and B-tile (BK x BN).
  // Pad by 1 to avoid bank conflicts on the K dimension.
  __shared__ scalar_t sA[BM][BK + 1];
  __shared__ scalar_t sB[BK][BN + 1];

  // Accumulator registers
  scalar_t acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; i++)
#pragma unroll
    for (int j = 0; j < TN; j++)
      acc[i][j] = 0.0f;

  // Number of K-tiles
  const int nk = (N + BK - 1) / BK;

  for (int t = 0; t < nk; t++) {
    // Cooperative load: 256 threads load BM*BK = 1024 elements of A
    // and BK*BN = 1024 elements of B, i.e. 4 loads each.
#pragma unroll
    for (int i = 0; i < (BM * BK) / 256; i++) {
      int idx = i * 256 + tid;
      int r = idx / BK;
      int c = idx % BK;
      int gr = by * BM + r;
      int gc = t * BK + c;
      sA[r][c] = (gr < M && gc < N) ? A[gr * N + gc] : 0.0f;
    }
#pragma unroll
    for (int i = 0; i < (BK * BN) / 256; i++) {
      int idx = i * 256 + tid;
      int r = idx / BN;
      int c = idx % BN;
      int gr = t * BK + r;
      int gc = bx * BN + c;
      sB[r][c] = (gr < N && gc < P) ? B[gr * P + gc] : 0.0f;
    }
    __syncthreads();

    // Compute: each thread does TM x TN outer products over BK
#pragma unroll
    for (int k = 0; k < BK; k++) {
      scalar_t ra[TM], rb[TN];
#pragma unroll
      for (int i = 0; i < TM; i++)
        ra[i] = sA[ty * TM + i][k];
#pragma unroll
      for (int j = 0; j < TN; j++)
        rb[j] = sB[k][tx * TN + j];
#pragma unroll
      for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++)
          acc[i][j] += ra[i] * rb[j];
    }
    __syncthreads();
  }

  // Write results back to global memory
#pragma unroll
  for (int i = 0; i < TM; i++) {
    int gr = row0 + i;
    if (gr < M) {
#pragma unroll
      for (int j = 0; j < TN; j++) {
        int gc = col0 + j;
        if (gc < P) {
          C[gr * P + gc] = acc[i][j];
        }
      }
    }
  }
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out,
            uint32_t M, uint32_t N, uint32_t P) {
  dim3 block(BN / TN, BM / TM, 1);  // 16 x 16 = 256 threads
  dim3 grid((P + BN - 1) / BN, (M + BM - 1) / BM, 1);
  MatmulKernel<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
}

__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out, size_t size, size_t redice_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    scalar_t tmp = a[gid * redice_size];
    for (size_t i = 1; i < redice_size; ++i) {
      tmp = max(tmp, a[gid * redice_size + i]);
    }
    out[gid] = tmp;
  }
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
  /// END SOLUTION
}

__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out, size_t size, size_t redice_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    scalar_t tmp = 0;
    for (size_t i = 0; i < redice_size; ++i) {
      tmp += a[gid * redice_size + i];
    }
    out[gid] = tmp;
  }
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
  /// END SOLUTION
}

}  // namespace cuda
}  // namespace uniti

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace uniti;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);
  m.def("ewise_sin", EwiseSin);
  m.def("ewise_cos", EwiseCos);

  m.def("arange", Arange);
  m.def("triu_mask", TriuMask);
  m.def("embedding_lookup", EmbeddingLookup);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
