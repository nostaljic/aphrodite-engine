#include <torch/extension.h>

void rotary_embedding_cpu(torch::Tensor &positions, torch::Tensor &query,
                          torch::Tensor &key, int head_size,
                          torch::Tensor &cos_sin_cache, bool is_neox);

void silu_and_mul_cpu(torch::Tensor &out, torch::Tensor &input);

void gelu_new_cpu(torch::Tensor &out, torch::Tensor &input);

void gelu_fast_cpu(torch::Tensor &out, torch::Tensor &input);

void paged_attention_v1_cpu(
    torch::Tensor &out, torch::Tensor &query, torch::Tensor &key_cache,
    torch::Tensor &value_cache, int num_kv_heads, float scale,
    torch::Tensor &block_tables, torch::Tensor &context_lens, int block_size,
    int max_context_len, const c10::optional<torch::Tensor> &alibi_slopes,
    const std::string &kv_cache_dtype);

void paged_attention_v2_cpu(
    torch::Tensor &out, torch::Tensor &exp_sums, torch::Tensor &max_logits,
    torch::Tensor &tmp_out, torch::Tensor &query, torch::Tensor &key_cache,
    torch::Tensor &value_cache, int num_kv_heads, float scale,
    torch::Tensor &block_tables, torch::Tensor &context_lens, int block_size,
    int max_context_len, const c10::optional<torch::Tensor> &alibi_slopes,
    const std::string &kv_cache_dtype);

void copy_blocks_cpu(
    std::vector<torch::Tensor> &key_caches,
    std::vector<torch::Tensor> &value_caches,
    const std::map<int64_t, std::vector<int64_t>> &block_mapping);

void reshape_and_cache_cpu(torch::Tensor &key, torch::Tensor &value,
                           torch::Tensor &key_cache, torch::Tensor &value_cache,
                           torch::Tensor &slot_mapping);

void swap_blocks_cpu(torch::Tensor &src, torch::Tensor &dst,
                     const std::map<int64_t, int64_t> &block_mapping);

void gather_cached_kv_cpu(torch::Tensor &key, torch::Tensor &value,
                          torch::Tensor &key_cache, torch::Tensor &value_cache,
                          torch::Tensor &slot_mapping);

void rms_norm_cpu(torch::Tensor &out, torch::Tensor &input,
                  torch::Tensor &weight, float epsilon);

void fused_add_rms_norm_cpu(torch::Tensor &input, torch::Tensor &residual,
                            torch::Tensor &weight, float epsilon);

inline torch::Tensor awq_gemm_cpu(torch::Tensor _in_feats, torch::Tensor _kernel,
                           torch::Tensor _scaling_factors, torch::Tensor _zeros,
                           int split_k_iters) {
  TORCH_CHECK(false, "Quantization is not supported on CPU.");
}

inline torch::Tensor awq_dequantize_cpu(torch::Tensor _kernel, torch::Tensor _scaling_factors,
                                        torch::Tensor _zeros, int split_k_iters, int thy,
                                        int thy) {
  TORCH_CHECK(false, "Quantization is not supported on CPU.");
}

inline void marlin_gemm_cpu(
  const torch::Tensor& input,
  const torch::Tensor& weights,
  torch::Tensor& output,
  const torch::Tensor& scales,
  torch::Tensor& workspace) {
  TORCH_CHECK(false, "Quantization is not supported on CPU.");
}

inline at::Tensor e8p_mm_origorder(
  const at::Tensor& A,
  const at::Tensor& B,
  const at::Tensor & CB) {
  TORCH_CHECK(false, "Quantization is not supported on CPU.");
}

inline void decompress_e8p_origorder(
  torch::Tensor YIs,
  torch::Tensor CB,
  torch::Tensor &Y) {
  TORCH_CHECK(false, "Quantization is not supported on CPU.");
}


inline void squeezellm_gemm_cpu(torch::Tensor vec, torch::Tensor mat,
                         torch::Tensor mul, torch::Tensor lookup_table) {
  TORCH_CHECK(false, "Quantization is not supported on CPU.");
}

inline torch::Tensor gptq_gemm_cpu(
  torch::Tensor a,
  torch::Tensor b_q_weight,
  torch::Tensor b_gptq_qzeros,
  torch::Tensor b_gptq_scales,
  torch::Tensor b_g_idx,
  bool use_exllama,
  int bit) {
  TORCH_CHECK(false, "Quantization is not supported on CPU.");
}

inline void gptq_shuffle_cpu(
  torch::Tensor q_weight,
  torch::Tensor q_perm,
  int bit) {
  TORCH_CHECK(false, "Quantization is not supported on CPU.");
}

inline torch::Tensor ggml_dequantize(
  torch::Tensor X,
  int8_t type,
  int64_t m,
  int64_t n) {
  TORCH_CHECK(false, "Quantization is not supported on CPU.");
}

inline torch::Tensor ggml_mul_mat_vec_cpu(
  torch::Tensor W,  // quant weight
  torch::Tensor X,  // input
  int8_t type,
  int64_t m) {
  TORCH_CHECK(false, "Quantization is not supported on CPU.");
}

inline torch::Tensor ggml_mul_mat_vec_a8_cpu(
  torch::Tensor W,  // quant weight
  torch::Tensor X,  // input
  int8_t type,
  int64_t row) {
  TORCH_CHECK(false, "Quantization is not supported on CPU.");
}

inline torch::Tensor ggml_mul_mat_a8_cpu(
  torch::Tensor W,  // quant weight
  torch::Tensor X,  // input
  int8_t type,
  int64_t row) {
  TORCH_CHECK(false, "Quantization is not supported on CPU.");
}
