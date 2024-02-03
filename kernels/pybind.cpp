#include "cache.h"
#include "cuda_utils.h"
#include "ops.h"
#include "cpu/cpu_ops.h"
#include "dispatch_utils.h"
#include <torch/extension.h>

void rotary_embedding_dispatch(torch::Tensor &positions, torch::Tensor &query,
                               torch::Tensor &key, int head_size,
                               torch::Tensor &cos_sin_cache, bool is_neox) {
  APHRODITE_DISPATCH_DEVICES(key.device(), rotary_embedding, positions, query, key, head_size, cos_sin_cache, is_neox);
}
void rotary_embedding_dispatch(torch::Tensor &positions, torch::Tensor &query,
                               torch::Tensor &key, int head_size,
                               torch::Tensor &cos_sin_cache, bool is_neox) {
  APHRODITE_DISPATCH_DEVICES(key.device(), rotary_embedding, positions, query, key, head_size, cos_sin_cache, is_neox);
}

void paged_attention_v1_dispatch(
    torch::Tensor &out, torch::Tensor &query, torch::Tensor &key_cache,
    torch::Tensor &value_cache, int num_kv_heads, float scale,
    torch::Tensor &block_tables, torch::Tensor &context_lens, int block_size,
    int max_context_len, const c10::optional<torch::Tensor> &alibi_slopes,
    const std::string &kv_cache_dtype) {
  APHRODITE_DISPATCH_DEVICES(out.device(), paged_attention_v1, out, query, key_cache, value_cache, num_kv_heads, scale, block_tables, context_lens, block_size, max_context_len, alibi_slopes, kv_cache_dtype);
}

void paged_attention_v2_dispatch(torch::Tensor &out, torch::Tensor &exp_sums,
    torch::Tensor &max_logits, torch::Tensor &tmp_out, torch::Tensor &query, 
    torch::Tensor &key_cache, torch::Tensor &value_cache, int num_kv_heads, 
    float scale, torch::Tensor &block_tables, torch::Tensor &context_lens, int block_size,
    int max_context_len, const c10::optional<torch::Tensor> &alibi_slopes,
    const std::string &kv_cache_dtype) {
  APHRODITE_DISPATCH_DEVICES(out.device(), paged_attention_v2, out, exp_sums, max_logits, tmp_out, query, key_cache, value_cache, num_kv_heads, scale, block_tables, context_lens, block_size,max_context_len, alibi_slopes, kv_cache_dtype);
}

void silu_and_mul_dispatch(torch::Tensor &out, torch::Tensor &input) {
  APHRODITE_DISPATCH_DEVICES(out.device(), silu_and_mul, out, input);
}

void gelu_new_dispatch(torch::Tensor &out, torch::Tensor &input) {
  APHRODITE_DISPATCH_DEVICES(out.device(), gelu_new, out, input);
}

void gelu_fast_dispatch(torch::Tensor &out, torch::Tensor &input) {
  APHRODITE_DISPATCH_DEVICES(out.device(), gelu_fast, out, input);
}

void rms_norm_dispatch(torch::Tensor &out, torch::Tensor &input, torch::Tensor &weight, float epsilon) {
  APHRODITE_DISPATCH_DEVICES(out.device(), rms_norm, out, input, weight, epsilon);
}

void fused_add_rms_norm_dispatch(torch::Tensor& input, torch::Tensor& residual, torch::Tensor& weight, float epsilon) {
  APHRODITE_DISPATCH_DEVICES(input.device(), fused_add_rms_norm, input, residual, weight, epsilon);
}

torch::Tensor awq_gemm_dispatch(torch::Tensor _in_feats, torch::Tensor _kernel, torch::Tensor _scaling_factors, torch::Tensor _zeros, int split_k_iters) {
  APHRODITE_DISPATCH_DEVICES(_in_feats.device(), awq_gemm, _in_feats, _kernel, _scaling_factors, _zeros, split_k_iters);
}

torch::Tensor awq_dequantize_dispatch(torch::Tensor _kernel, torch::Tensor _scaling_factors, torch::Tensor _zeros, int split_k_iters, int thx, int thy) {
  APHRODITE_DISPATCH_DEVICES(_kernel.device(), awq_dequantize, _kernel, _scaling_factors, _zeros, split_k_iters, thx, thy);
}

void squeezellm_gemm_dispatch(torch::Tensor vec, torch::Tensor mat, torch::Tensor mul, torch::Tensor lookup_table) {
  APHRODITE_DISPATCH_DEVICES(vec.device(), squeezellm_gemm, vec, mat, mul, lookup_table);
}

void marlin_gemm_dispatch(const torch::Tensor& input, const torch::Tensor& weights, torch::Tensor& outputs, const torch::Tensor& scales, torch::Tensor& workspace) {
  APHRODITE_DISPATCH_DEVICES(input.device(), marlin_gemm, input, weights, outputs, scales, workspace);
}

at::Tensor e8p_mm_origorder_dispatch(const at::Tensor& A, const at::Tensor& B, const at::Tensor& CB) {
  APHRODITE_DISPATCH_DEVICES(A.device(), e8p_mm_origorder, A, B, CB);
}

void decompress_e8p_origorder_dispatch(torch::Tensor YIs, torch::Tensor CB, torch::Tensor &Y) {
  APHRODITE_DISPATCH_DEVICES(YIs.device(), decompress_e8p_origorder, YIs, CB, Y);
}

void swap_blocks_dispatch_dispatch(torch::Tensor& src, torch::Tensor& dst, const std::map<int64_t, int64_t>& block_mapping) {
  APHRODITE_DISPATCH_DEVICES(src.device(), swap_blocks, src, dst, block_mapping);
}

void copy_blocks_dispatch(std::vector<torch::Tensor>& key_caches, std::vector<torch::Tensor>& value_caches, const std::map<int64_t, std::vector<int64_t>>& block_mapping) {
  APHRODITE_DISPATCH_DEVICES(key_caches[0].device(), copy_blocks, key_caches, value_caches, block_mapping);
}

void reshape_and_cache_dispatch(torch::Tensor& key, torch::Tensor& value, torch::Tensor& key_cache, torch::Tensor& value_cache, torch::Tensor& slot_mapping, const std::string& kv_cache_dtype) {
  APHRODITE_DISPATCH_DEVICES(key.device(), reshape_and_cache, key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype);
}


void gather_cached_kv_dispatch(torch::Tensor& key, torch::Tensor& value, torch::Tensor& key_cache, torch::Tensor& value_cache, torch::Tensor& slot_mapping) {
  APHRODITE_DISPATCH_DEVICES(key.device(), gather_cached_kv, key, value, key_cache, value_cache, slot_mapping);
}

void convert_fp8_e5m2_dispatch(
  torch::Tensor& src_cache,
  torch::Tensor& dst_cache) {
    APHRODITE_DISPATCH_DEVICES(src_cache.device(), convert_fp8_e5m2, src_cache, dst_cache);
}

torch::Tensor gptq_gemm_dispatch(
  torch::Tensor a,
  torch::Tensor b_q_weight,
  torch::Tensor b_gptq_qzeros,
  torch::Tensor b_gptq_scales,
  torch::Tensor b_g_idx,
  bool use_exllama,
  int bit) {
    APHRODITE_DISPATCH_DEVICES(a.device(), gptq_gemm, a, b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx, use_exllama, bit);
}

void gptq_shuffle_dispatch(
  torch::Tensor q_weight,
  torch::Tensor q_perm,
  int bit) {
    APHRODITE_DISPATCH_DEVICES(q_weight.device(), gptq_shuffle, q_weight, q_perm, bit);
}

torch::Tensor ggml_dequantize_dispatch(
  torch::Tensor X,
  int8_t type,
  int64_t m,
  int64_t n) {
    APHRODITE_DISPATCH_DEVICES(X.device(), ggml_dequantize, X, type, m, n);
}

torch::Tensor ggml_mul_mat_vec_dispatch(
  torch::Tensor W,
  torch::Tensor X,
  int8_t type,
  int64_t m) {
    APHRODITE_DISPATCH_DEVICES(W.device(), ggml_mul_mat_vec, W, X, type, m);
}

torch::Tensor ggml_mul_mat_vec_a8_dispatch(
  torch::Tensor W,
  torch::Tensor X,
  int8_t type,
  int64_t row) {
    APHRODITE_DISPATCH_DEVICES(W.device(), ggml_mul_mat_vec_a8, W, X, type, row);
}

torch::Tensor ggml_mul_mat_a8_dispatch(
  torch::Tensor W,
  torch::Tensor X,
  int8_t type,
  int64_t row) {
    APHRODITE_DISPATCH_DEVICES(W.device(), ggml_mul_mat_a8, W, X, type, row);
}

// using fptr_t = uint64_t;

// fptr_t init_custom_ar_dispatch(
//   torch::Tensor &meta,
//   torch::Tensor &rank_data,
//   const std::vector<std::string> &handles,
//   const std::vector<int64_t> &offsets,
//   int rank,
//   bool full_nvlink) {
//     APHRODITE_DISPATCH_DEVICES(meta.device(), init_custom_ar, meta, rank_data, handles, offsets, rank, full_nvlink);
// }

// bool should_custom_ar_dispatch(
//   torch::Tensor &inp,
//   int max_size,
//   int world_size,
//   bool full_nvlink) {
//     APHRODITE_DISPATCH_DEVICES(inp.device(), should_custom_ar, inp, max_size, world_size, full_nvlink);
// }

// void all_reduce_reg_dispatch(
//   fptr_t _fa,
//   torch::Tensor &inp,
//   torch::Tensor &out) {
//     APHRODITE_DISPATCH_DEVICES(inp.device(), all_reduce_reg, _fa, inp, out);
// }

#ifdef APHRODITE_BUILD_CPU_ONLY
int get_device_attribute(
    int attribute,
    int device_id) { return 94387; }
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Aphrodite custom ops
  pybind11::module ops = m.def_submodule("ops", "Aphrodite custom operators");

  // Attention ops
  ops.def(
    "paged_attention_v1",
    &paged_attention_v1_dispatch,
    "Compute the attention between an input query and the cached keys/values using PagedAttention.");
  ops.def(
    "paged_attention_v2",
    &paged_attention_v2_dispatch,
    "PagedAttention V2.");

  // Activation ops
  ops.def(
    "silu_and_mul",
    &silu_and_mul_dispatch,
    "Activation function used in SwiGLU.");
  ops.def(
    "gelu_new",
    &gelu_new_dispatch,
    "GELU implementation used in GPT-2.");
  ops.def(
    "gelu_fast",
    &gelu_fast_dispatch,
    "Approximate GELU implementation.");

  // Layernorm
  ops.def(
    "rms_norm",
    &rms_norm_dispatch,
    "Apply Root Mean Square (RMS) Normalization to the input tensor.");

  ops.def(
    "fused_add_rms_norm",
    &fused_add_rms_norm_dispatch,
    "In-place fused Add and RMS Normalization");

  // Rotary embedding
  ops.def(
    "rotary_embedding",
    &rotary_embedding_dispatch,
    "Apply GPT-NeoX or GPT-J style rotary embedding to query and key");

#ifndef USE_ROCM
  // Quantization ops
  ops.def("awq_gemm", &awq_gemm_dispatch, "Quantized GEMM for AWQ");
  ops.def("awq_dequantize", &awq_dequantize_dispatch, "Dequantization for AWQ");
  ops.def("quip_decompress", &decompress_e8p_origorder_dispatch, "decompress_packed_e8p");
  ops.def("quip_gemv", &e8p_mm_origorder_dispatch, "e8p_mm_origorder");
  ops.def("marlin_gemm", &marlin_gemm_dispatch, "Marlin Optimized Quantized GEMM for GPTQ");
#endif
  ops.def("gptq_gemm", &gptq_gemm_dispatch, "Quantized GEMM for GPTQ");
  ops.def("gptq_shuffle", &gptq_shuffle_dispatch, "Post processing for GPTQ");
  ops.def("squeezellm_gemm", &squeezellm_gemm_dispatch, "Quantized GEMM for SqueezeLLM");
  ops.def("ggml_dequantize", &ggml_dequantize_dispatch, "ggml_dequantize");
  ops.def("ggml_mul_mat_vec", &ggml_mul_mat_vec_dispatch, "ggml_mul_mat_vec");
  ops.def("ggml_mul_mat_vec_a8", &ggml_mul_mat_vec_a8_dispatch, "ggml_mul_mat_vec_a8");
  ops.def("ggml_mul_mat_a8", &ggml_mul_mat_a8_dispatch, "ggml_mul_mat_a8");
  
  ops.def("moe_align_block_size",
          &moe_align_block_size,
          "Aligning the number of tokens to be processed by each expert such that it is divisible by the block size.");

  // Cache ops
  pybind11::module cache_ops = m.def_submodule("cache_ops", "Aphrodite cache ops");
  cache_ops.def(
    "swap_blocks",
    &swap_blocks_dispatch_dispatch,
    "Swap in (out) the cache blocks from src to dst");
  cache_ops.def(
    "copy_blocks",
    &copy_blocks_dispatch,
    "Copy the cache blocks from src to dst");
  cache_ops.def(
    "reshape_and_cache",
    &reshape_and_cache_dispatch,
    "Reshape the key and value tensors and cache them");
  cache_ops.def(
    "gather_cached_kv",
    &gather_cached_kv_dispatch,
    "Gather key and value from the cache into contiguous QKV tensors");
  cache_ops.def(
    "convert_fp8_e5m2",
    &convert_fp8_e5m2_dispatch,
    "Convert the key and value cache to fp8_e5m2 data type");

  // Cuda utils
  pybind11::module cuda_utils = m.def_submodule("cuda_utils", "Aphrodite cuda utils");
  cuda_utils.def(
    "get_device_attribute",
    &get_device_attribute,
    "Gets the specified device attribute.");

  cuda_utils.def(
    "get_max_shared_memory_per_block_device_attribute",
    &get_max_shared_memory_per_block_device_attribute,
    "Gets the maximum shared memory per block device attribute.");

#ifndef USE_ROCM
  // Custom all-reduce kernels
  pybind11::module custom_ar = m.def_submodule("custom_ar", "custom allreduce");
  custom_ar.def("init_custom_ar", &init_custom_ar, "init_custom_ar");
  custom_ar.def("should_custom_ar", &should_custom_ar, "should_custom_ar");
  custom_ar.def("all_reduce_reg", &all_reduce_reg, "all_reduce_reg");
  custom_ar.def("all_reduce_unreg", &all_reduce_unreg, "all_reduce_unreg");
  custom_ar.def("dispose", &dispose, "dispose");
  custom_ar.def("meta_size", &meta_size, "meta_size");
  custom_ar.def("register_buffer", &register_buffer, "register_buffer");
  custom_ar.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta,
                "get_graph_buffer_ipc_meta");
  custom_ar.def("register_graph_buffers", &register_graph_buffers,
                "register_graph_buffers");
#endif

}
