/**
 * Apple Metal GPU backend for UniTI framework.
 *
 * Architecture mirrors ndarray_backend_cuda.cu:
 *   - MetalArray holds an MTLBuffer (data lives on GPU)
 *   - Compute kernels dispatched via MTLComputeCommandEncoder
 *   - from_numpy / to_numpy transfer data between Host and GPU
 *   - On Apple Silicon, MTLBuffer uses shared memory (zero-copy!)
 *
 * Build: compiled by CMake as Objective-C++ (.mm) with -framework Metal -framework Foundation
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <string>
#include <unordered_map>

namespace uniti {
namespace metal {

typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

// Global Metal device and command queue (initialized once)
static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLLibrary> g_library = nil;
static std::unordered_map<std::string, id<MTLComputePipelineState>> g_pipelines;

// ──────────────────────────────────────────────────────────────────
// Metal Shader Library (embedded MSL source)
// ──────────────────────────────────────────────────────────────────
static const char* SHADER_SOURCE = R"(
#include <metal_stdlib>
using namespace metal;

kernel void ewise_add(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]], uint id [[thread_position_in_grid]]) { out[id] = a[id] + b[id]; }
kernel void ewise_mul(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]], uint id [[thread_position_in_grid]]) { out[id] = a[id] * b[id]; }
kernel void ewise_div(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]], uint id [[thread_position_in_grid]]) { out[id] = a[id] / b[id]; }
kernel void ewise_maximum(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]], uint id [[thread_position_in_grid]]) { out[id] = max(a[id], b[id]); }
kernel void ewise_eq(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]], uint id [[thread_position_in_grid]]) { out[id] = (a[id] == b[id]) ? 1.0f : 0.0f; }
kernel void ewise_ge(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]], uint id [[thread_position_in_grid]]) { out[id] = (a[id] >= b[id]) ? 1.0f : 0.0f; }

kernel void scalar_add(device const float* a [[buffer(0)]], device const float* v [[buffer(1)]], device float* out [[buffer(2)]], uint id [[thread_position_in_grid]]) { out[id] = a[id] + v[0]; }
kernel void scalar_mul(device const float* a [[buffer(0)]], device const float* v [[buffer(1)]], device float* out [[buffer(2)]], uint id [[thread_position_in_grid]]) { out[id] = a[id] * v[0]; }
kernel void scalar_div(device const float* a [[buffer(0)]], device const float* v [[buffer(1)]], device float* out [[buffer(2)]], uint id [[thread_position_in_grid]]) { out[id] = a[id] / v[0]; }
kernel void scalar_power(device const float* a [[buffer(0)]], device const float* v [[buffer(1)]], device float* out [[buffer(2)]], uint id [[thread_position_in_grid]]) { out[id] = pow(a[id], v[0]); }
kernel void scalar_maximum(device const float* a [[buffer(0)]], device const float* v [[buffer(1)]], device float* out [[buffer(2)]], uint id [[thread_position_in_grid]]) { out[id] = max(a[id], v[0]); }
kernel void scalar_eq(device const float* a [[buffer(0)]], device const float* v [[buffer(1)]], device float* out [[buffer(2)]], uint id [[thread_position_in_grid]]) { out[id] = (a[id] == v[0]) ? 1.0f : 0.0f; }
kernel void scalar_ge(device const float* a [[buffer(0)]], device const float* v [[buffer(1)]], device float* out [[buffer(2)]], uint id [[thread_position_in_grid]]) { out[id] = (a[id] >= v[0]) ? 1.0f : 0.0f; }

kernel void ewise_log(device const float* a [[buffer(0)]], device float* out [[buffer(1)]], uint id [[thread_position_in_grid]]) { out[id] = log(a[id]); }
kernel void ewise_exp(device const float* a [[buffer(0)]], device float* out [[buffer(1)]], uint id [[thread_position_in_grid]]) { out[id] = exp(a[id]); }
kernel void ewise_tanh(device const float* a [[buffer(0)]], device float* out [[buffer(1)]], uint id [[thread_position_in_grid]]) { out[id] = tanh(a[id]); }
kernel void ewise_sin(device const float* a [[buffer(0)]], device float* out [[buffer(1)]], uint id [[thread_position_in_grid]]) { out[id] = sin(a[id]); }
kernel void ewise_cos(device const float* a [[buffer(0)]], device float* out [[buffer(1)]], uint id [[thread_position_in_grid]]) { out[id] = cos(a[id]); }

kernel void fill_kernel(device const float* v [[buffer(0)]], device float* out [[buffer(1)]], uint id [[thread_position_in_grid]]) { out[id] = v[0]; }

kernel void compact_kernel(device const float* a [[buffer(0)]], device float* out [[buffer(1)]],
                           device const int32_t* shape [[buffer(2)]], device const int32_t* strides [[buffer(3)]],
                           device const uint* params [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    uint ndim = params[0];
    uint offset = params[1];
    uint cnt = params[2];
    if (gid >= cnt) return;
    // Same index conversion as CUDA GetOriIndex
    uint cur_size = 1, pre_size = 1;
    uint src_pos = offset;
    for (int i = (int)ndim - 1; i >= 0; i--) {
        cur_size *= (uint)shape[i];
        uint dim_idx = (gid % cur_size) / pre_size;
        pre_size = cur_size;
        src_pos += dim_idx * (uint)strides[i];
    }
    out[gid] = a[src_pos];
}

kernel void ewise_setitem_kernel(device const float* a [[buffer(0)]], device float* out [[buffer(1)]],
                                  device const int32_t* shape [[buffer(2)]], device const int32_t* strides [[buffer(3)]],
                                  device const uint* params [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    uint ndim = params[0];
    uint offset = params[1];
    uint cnt = params[2];
    if (gid >= cnt) return;
    uint cur_size = 1, pre_size = 1;
    uint dst_pos = offset;
    for (int i = (int)ndim - 1; i >= 0; i--) {
        cur_size *= (uint)shape[i];
        uint dim_idx = (gid % cur_size) / pre_size;
        pre_size = cur_size;
        dst_pos += dim_idx * (uint)strides[i];
    }
    out[dst_pos] = a[gid];
}

kernel void scalar_setitem_kernel(device float* out [[buffer(0)]],
                                   device const float* v [[buffer(1)]],
                                   device const int32_t* shape [[buffer(2)]], device const int32_t* strides [[buffer(3)]],
                                   device const uint* params [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    uint ndim = params[0];
    uint offset = params[1];
    uint cnt = params[2];
    if (gid >= cnt) return;
    uint cur_size = 1, pre_size = 1;
    uint dst_pos = offset;
    for (int i = (int)ndim - 1; i >= 0; i--) {
        cur_size *= (uint)shape[i];
        uint dim_idx = (gid % cur_size) / pre_size;
        pre_size = cur_size;
        dst_pos += dim_idx * (uint)strides[i];
    }
    out[dst_pos] = v[0];
}

kernel void matmul_kernel(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                          device float* out [[buffer(2)]], device const uint* p [[buffer(3)]],
                          uint id [[thread_position_in_grid]]) {
    uint M = p[0], N = p[1], P = p[2];
    if (id >= M * P) return;
    uint i = id / P, j = id % P;
    float s = 0.0f;
    for (uint k = 0; k < N; k++) s += a[i * N + k] * b[k * P + j];
    out[id] = s;
}

kernel void reduce_sum_kernel(device const float* a [[buffer(0)]], device float* out [[buffer(1)]],
                              device const uint* p [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    uint rs = p[0]; float s = 0.0f; uint base = id * rs;
    for (uint i = 0; i < rs; i++) s += a[base + i];
    out[id] = s;
}

kernel void reduce_max_kernel(device const float* a [[buffer(0)]], device float* out [[buffer(1)]],
                              device const uint* p [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    uint rs = p[0]; uint base = id * rs; float m = a[base];
    for (uint i = 1; i < rs; i++) m = max(m, a[base + i]);
    out[id] = m;
}

kernel void arange_kernel(device float* out [[buffer(0)]], uint id [[thread_position_in_grid]]) {
    out[id] = float(id);
}

kernel void triu_mask_kernel(device float* out [[buffer(0)]], device const uint* p [[buffer(1)]],
                             device const float* mv [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    uint cols = p[0]; uint k_val = p[1];
    uint i = gid / cols, j = gid % cols;
    out[gid] = (j >= i + k_val) ? mv[0] : 0.0f;
}

kernel void embedding_lookup_kernel(device const float* weight [[buffer(0)]], device const float* ids [[buffer(1)]],
                                     device float* out [[buffer(2)]], device const uint* p [[buffer(3)]],
                                     uint gid [[thread_position_in_grid]]) {
    uint embedding_dim = p[0];
    uint token_idx = gid / embedding_dim;
    uint dim_idx = gid % embedding_dim;
    int vocab_idx = int(ids[token_idx]);
    out[gid] = weight[vocab_idx * embedding_dim + dim_idx];
}
)";

// ──────────────────────────────────────────────────────────────────
// Initialization
// ──────────────────────────────────────────────────────────────────
void init_metal() {
    if (g_device) return;
    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) throw std::runtime_error("No Metal device found");
    g_queue = [g_device newCommandQueue];

    NSError* error = nil;
    NSString* src = [NSString stringWithUTF8String:SHADER_SOURCE];
    g_library = [g_device newLibraryWithSource:src options:nil error:&error];
    if (!g_library) {
        throw std::runtime_error(std::string("Metal shader compile error: ") +
                                 [[error localizedDescription] UTF8String]);
    }
}

id<MTLComputePipelineState> get_pipeline(const std::string& name) {
    auto it = g_pipelines.find(name);
    if (it != g_pipelines.end()) return it->second;

    id<MTLFunction> fn = [g_library newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
    if (!fn) throw std::runtime_error("Metal function not found: " + name);

    NSError* error = nil;
    id<MTLComputePipelineState> pso = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pso) throw std::runtime_error("Pipeline creation failed: " + name);

    g_pipelines[name] = pso;
    return pso;
}

// ──────────────────────────────────────────────────────────────────
// Dispatch helper
// ──────────────────────────────────────────────────────────────────
void dispatch(const std::string& name, size_t grid_size, std::vector<id<MTLBuffer>> buffers) {
    auto pso = get_pipeline(name);
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];
    for (size_t i = 0; i < buffers.size(); i++) {
        [enc setBuffer:buffers[i] offset:0 atIndex:i];
    }
    // Use dispatchThreads for exact thread count (no over-dispatch).
    // Available on all Apple Silicon (GPU Family Apple 4+).
    NSUInteger threadGroupSize = std::min((NSUInteger)256, pso.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake(grid_size, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threadGroupSize, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

// Helper: create a small buffer with scalar value
id<MTLBuffer> scalar_buffer(scalar_t val) {
    id<MTLBuffer> buf = [g_device newBufferWithLength:ELEM_SIZE options:MTLResourceStorageModeShared];
    *(scalar_t*)buf.contents = val;
    return buf;
}

// Helper: create a buffer from uint32 vector
id<MTLBuffer> uint_buffer(const std::vector<uint32_t>& data) {
    id<MTLBuffer> buf = [g_device newBufferWithLength:data.size() * sizeof(uint32_t) options:MTLResourceStorageModeShared];
    memcpy(buf.contents, data.data(), data.size() * sizeof(uint32_t));
    return buf;
}

// Helper: create a buffer from int32 vector
id<MTLBuffer> int_buffer(const std::vector<int32_t>& data) {
    id<MTLBuffer> buf = [g_device newBufferWithLength:data.size() * sizeof(int32_t) options:MTLResourceStorageModeShared];
    memcpy(buf.contents, data.data(), data.size() * sizeof(int32_t));
    return buf;
}

// ──────────────────────────────────────────────────────────────────
// MetalArray — mirrors CudaArray
// ──────────────────────────────────────────────────────────────────
struct MetalArray {
    MetalArray(size_t size) : size(size) {
        init_metal();
        // Apple Silicon shared memory = zero-copy between CPU & GPU
        buf = [g_device newBufferWithLength:size * ELEM_SIZE options:MTLResourceStorageModeShared];
        if (!buf) throw std::runtime_error("Metal buffer allocation failed");
    }
    ~MetalArray() { buf = nil; }

    scalar_t* ptr() { return (scalar_t*)buf.contents; }
    size_t ptr_as_int() { return (size_t)buf.contents; }

    id<MTLBuffer> buf;
    size_t size;
};

// ──────────────────────────────────────────────────────────────────
// NDArray API functions
// ──────────────────────────────────────────────────────────────────

void Fill(MetalArray* out, scalar_t val) {
    dispatch("fill_kernel", out->size, {scalar_buffer(val), out->buf});
}

void Compact(const MetalArray& a, MetalArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
    size_t cnt = 1;
    for (auto s : shape) cnt *= s;
    uint32_t ndim = shape.size();
    auto params = uint_buffer({ndim, (uint32_t)offset, (uint32_t)cnt});
    dispatch("compact_kernel", cnt, {a.buf, out->buf, int_buffer(shape), int_buffer(strides), params});
}

void EwiseSetitem(const MetalArray& a, MetalArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
    size_t cnt = 1;
    for (auto s : shape) cnt *= s;
    uint32_t ndim = shape.size();
    auto params = uint_buffer({ndim, (uint32_t)offset, (uint32_t)cnt});
    dispatch("ewise_setitem_kernel", cnt, {a.buf, out->buf, int_buffer(shape), int_buffer(strides), params});
}

void ScalarSetitem(size_t size, scalar_t val, MetalArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
    size_t cnt = 1;
    for (auto s : shape) cnt *= s;
    uint32_t ndim = shape.size();
    auto params = uint_buffer({ndim, (uint32_t)offset, (uint32_t)cnt});
    dispatch("scalar_setitem_kernel", cnt, {out->buf, scalar_buffer(val), int_buffer(shape), int_buffer(strides), params});
}

// Ewise binary ops
#define EWISE_BINARY(Name, kernel_name) \
void Name(const MetalArray& a, const MetalArray& b, MetalArray* out) { \
    dispatch(kernel_name, out->size, {a.buf, b.buf, out->buf}); \
}
EWISE_BINARY(EwiseAdd, "ewise_add")
EWISE_BINARY(EwiseMul, "ewise_mul")
EWISE_BINARY(EwiseDiv, "ewise_div")
EWISE_BINARY(EwiseMaximum, "ewise_maximum")
EWISE_BINARY(EwiseEq, "ewise_eq")
EWISE_BINARY(EwiseGe, "ewise_ge")

// Scalar ops
#define SCALAR_OP(Name, kernel_name) \
void Name(const MetalArray& a, scalar_t val, MetalArray* out) { \
    dispatch(kernel_name, out->size, {a.buf, scalar_buffer(val), out->buf}); \
}
SCALAR_OP(ScalarAdd, "scalar_add")
SCALAR_OP(ScalarMul, "scalar_mul")
SCALAR_OP(ScalarDiv, "scalar_div")
SCALAR_OP(ScalarPower, "scalar_power")
SCALAR_OP(ScalarMaximum, "scalar_maximum")
SCALAR_OP(ScalarEq, "scalar_eq")
SCALAR_OP(ScalarGe, "scalar_ge")

// Unary math
#define EWISE_UNARY(Name, kernel_name) \
void Name(const MetalArray& a, MetalArray* out) { \
    dispatch(kernel_name, out->size, {a.buf, out->buf}); \
}
EWISE_UNARY(EwiseLog, "ewise_log")
EWISE_UNARY(EwiseExp, "ewise_exp")
EWISE_UNARY(EwiseTanh, "ewise_tanh")
EWISE_UNARY(EwiseSin, "ewise_sin")
EWISE_UNARY(EwiseCos, "ewise_cos")

void Matmul(const MetalArray& a, const MetalArray& b, MetalArray* out,
            uint32_t m, uint32_t n, uint32_t p) {
    dispatch("matmul_kernel", m * p, {a.buf, b.buf, out->buf, uint_buffer({m, n, p})});
}

void ReduceMax(const MetalArray& a, MetalArray* out, size_t reduce_size) {
    dispatch("reduce_max_kernel", out->size, {a.buf, out->buf, uint_buffer({(uint32_t)reduce_size})});
}

void ReduceSum(const MetalArray& a, MetalArray* out, size_t reduce_size) {
    dispatch("reduce_sum_kernel", out->size, {a.buf, out->buf, uint_buffer({(uint32_t)reduce_size})});
}

void Arange(MetalArray* out, size_t n) {
    dispatch("arange_kernel", n, {out->buf});
}

void TriuMask(MetalArray* out, size_t rows, size_t cols, int k, scalar_t mask_val) {
    size_t total = rows * cols;
    dispatch("triu_mask_kernel", total, {out->buf, uint_buffer({(uint32_t)cols, (uint32_t)k}), scalar_buffer(mask_val)});
}

void EmbeddingLookup(const MetalArray& weight, const MetalArray& ids, MetalArray* out,
                     size_t num_ids, size_t embedding_dim) {
    size_t total = num_ids * embedding_dim;
    dispatch("embedding_lookup_kernel", total, {weight.buf, ids.buf, out->buf, uint_buffer({(uint32_t)embedding_dim})});
}

}  // namespace metal
}  // namespace uniti

// ──────────────────────────────────────────────────────────────────
// Pybind11 module — mirrors ndarray_backend_cuda exactly
// ──────────────────────────────────────────────────────────────────
PYBIND11_MODULE(ndarray_backend_metal, m) {
    namespace py = pybind11;
    using namespace uniti;
    using namespace metal;

    m.attr("__device_name__") = "metal";
    m.attr("__tile_size__") = 4;

    py::class_<MetalArray>(m, "Array")
        .def(py::init<size_t>(), py::return_value_policy::take_ownership)
        .def_readonly("size", &MetalArray::size)
        .def("ptr", &MetalArray::ptr_as_int);

    // return numpy array — zero-copy on Apple Silicon (shared memory)
    m.def("to_numpy", [](const MetalArray& a, std::vector<size_t> shape,
                         std::vector<size_t> strides, size_t offset) {
        std::vector<size_t> numpy_strides = strides;
        std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                       [](size_t& c) { return c * ELEM_SIZE; });
        return py::array_t<scalar_t>(shape, numpy_strides, (scalar_t*)a.buf.contents + offset);
    });

    // copy from numpy
    m.def("from_numpy", [](py::array_t<scalar_t> a, MetalArray* out) {
        std::memcpy(out->buf.contents, a.request().ptr, out->size * ELEM_SIZE);
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
