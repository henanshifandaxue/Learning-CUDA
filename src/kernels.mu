#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

#include "../tester/utils.h"

// ==================== Trace Function ====================

template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
    if (h_input.empty() || rows == 0 || cols == 0) {
        return T(0);
    }
    
    if (h_input.size() != rows * cols) {
        std::cerr << "Error: Input size doesn't match rows*cols" << std::endl;
        return T(-1);
    }
    
    T sum = T(0);
    size_t min_dim = std::min(rows, cols);
    
    for (size_t i = 0; i < min_dim; ++i) {
        sum += h_input[i * cols + i];
    }
    
    return sum;
}

// ==================== Half Type Support ====================

// 简单的 half 类型定义，实际应根据摩尔平台的头文件调整
struct half {
    unsigned short x;
    
    half() : x(0) {}
    half(float f) { fromFloat(f); }
    
    void fromFloat(float f) {
        // 简化版的 float 到 half 转换
        unsigned int i = *reinterpret_cast<unsigned int*>(&f);
        int s = (i >> 31) & 0x1;
        int e = (i >> 23) & 0xFF;
        int m = i & 0x7FFFFF;
        
        int out_e = 0;
        int out_m = 0;
        
        if (e == 0xFF) { // NaN or Inf
            out_e = 0x1F;
            out_m = m ? 0x200 : 0;
        } else if (e == 0) { // Zero or denormal
            out_e = 0;
            out_m = 0;
        } else {
            // Normal number
            int new_e = e - 127 + 15;
            if (new_e >= 31) { // Overflow
                out_e = 0x1F;
                out_m = 0;
            } else if (new_e <= 0) { // Underflow
                out_e = 0;
                out_m = 0;
            } else {
                out_e = new_e;
                out_m = m >> 13;
            }
        }
        
        x = (s << 15) | (out_e << 10) | out_m;
    }
    
    float toFloat() const {
        int s = (x >> 15) & 0x1;
        int e = (x >> 10) & 0x1F;
        int m = x & 0x3FF;
        
        if (e == 0x1F) {
            return (s ? -1.0f : 1.0f) * (m ? std::numeric_limits<float>::quiet_NaN() 
                                          : std::numeric_limits<float>::infinity());
        } else if (e == 0) {
            return s ? -0.0f : 0.0f;
        } else {
            int out_e = e - 15 + 127;
            unsigned int out_m = m << 13;
            unsigned int i = (s << 31) | (out_e << 23) | out_m;
            return *reinterpret_cast<float*>(&i);
        }
    }
    
    operator float() const { return toFloat(); }
};

// ==================== Flash Attention Functions ====================

template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
    
    // Validate input
    if (batch_size <= 0 || target_seq_len <= 0 || src_seq_len <= 0 ||
        query_heads <= 0 || kv_heads <= 0 || head_dim <= 0) {
        h_o.clear();
        return;
    }
    
    // Check GQA compatibility
    if (query_heads % kv_heads != 0) {
        kv_heads = query_heads;
    }
    
    // Calculate sizes
    size_t q_size = static_cast<size_t>(batch_size) * target_seq_len * query_heads * head_dim;
    size_t kv_size = static_cast<size_t>(batch_size) * src_seq_len * kv_heads * head_dim;
    size_t output_size = q_size;
    
    // Verify input sizes
    if (h_q.size() != q_size || h_k.size() != kv_size || h_v.size() != kv_size) {
        std::cerr << "Error: Input tensor sizes don't match expected dimensions" << std::endl;
        h_o.clear();
        return;
    }
    
    // Allocate output
    h_o.resize(output_size);
    
    // Calculate scaling factor
    float scale_factor = 1.0f / sqrtf(static_cast<float>(head_dim));
    int head_repeat = query_heads / kv_heads;
    
    // CPU reference implementation for 摩尔平台
    // 在实际部署中应使用摩尔平台的GPU内核
    for (int b = 0; b < batch_size; ++b) {
        for (int tgt_i = 0; tgt_i < target_seq_len; ++tgt_i) {
            for (int qh = 0; qh < query_heads; ++qh) {
                // GQA mapping
                int kvh = qh / head_repeat;
                if (kvh >= kv_heads) kvh = kv_heads - 1;
                
                // Temporary arrays
                std::vector<float> scores(src_seq_len, 0.0f);
                std::vector<float> attention_weights(src_seq_len, 0.0f);
                
                // Compute scores
                for (int src_j = 0; src_j < src_seq_len; ++src_j) {
                    // Apply causal mask
                    if (is_causal && tgt_i < src_j) {
                        scores[src_j] = -1e30f;
                        continue;
                    }
                    
                    float dot_product = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        size_t q_idx = ((b * target_seq_len + tgt_i) * query_heads + qh) * head_dim + d;
                        size_t k_idx = ((b * src_seq_len + src_j) * kv_heads + kvh) * head_dim + d;
                        
                        // Convert to float for computation
                        float q_val, k_val;
                        if (std::is_same<T, float>::value) {
                            q_val = h_q[q_idx];
                            k_val = h_k[k_idx];
                        } else if (std::is_same<T, half>::value) {
                            q_val = static_cast<half>(h_q[q_idx]).toFloat();
                            k_val = static_cast<half>(h_k[k_idx]).toFloat();
                        } else {
                            q_val = static_cast<float>(h_q[q_idx]);
                            k_val = static_cast<float>(h_k[k_idx]);
                        }
                        
                        dot_product += q_val * k_val;
                    }
                    
                    scores[src_j] = dot_product * scale_factor;
                }
                
                // Softmax
                float max_score = scores[0];
                for (int src_j = 1; src_j < src_seq_len; ++src_j) {
                    if (scores[src_j] > max_score) {
                        max_score = scores[src_j];
                    }
                }
                
                float exp_sum = 0.0f;
                for (int src_j = 0; src_j < src_seq_len; ++src_j) {
                    if (scores[src_j] < -1e29f) {
                        attention_weights[src_j] = 0.0f;
                    } else {
                        attention_weights[src_j] = expf(scores[src_j] - max_score);
                        exp_sum += attention_weights[src_j];
                    }
                }
                
                // Normalize
                if (exp_sum > 1e-12f) {
                    float inv_exp_sum = 1.0f / exp_sum;
                    for (int src_j = 0; src_j < src_seq_len; ++src_j) {
                        attention_weights[src_j] *= inv_exp_sum;
                    }
                }
                
                // Compute output
                for (int d = 0; d < head_dim; ++d) {
                    float weighted_sum = 0.0f;
                    
                    for (int src_j = 0; src_j < src_seq_len; ++src_j) {
                        if (attention_weights[src_j] > 0.0f) {
                            size_t v_idx = ((b * src_seq_len + src_j) * kv_heads + kvh) * head_dim + d;
                            float v_val;
                            
                            if (std::is_same<T, float>::value) {
                                v_val = h_v[v_idx];
                            } else if (std::is_same<T, half>::value) {
                                v_val = static_cast<half>(h_v[v_idx]).toFloat();
                            } else {
                                v_val = static_cast<float>(h_v[v_idx]);
                            }
                            
                            weighted_sum += attention_weights[src_j] * v_val;
                        }
                    }
                    
                    size_t o_idx = ((b * target_seq_len + tgt_i) * query_heads + qh) * head_dim + d;
                    
                    if (std::is_same<T, float>::value) {
                        h_o[o_idx] = weighted_sum;
                    } else if (std::is_same<T, half>::value) {
                        h_o[o_idx] = half(weighted_sum);
                    } else {
                        h_o[o_idx] = static_cast<T>(weighted_sum);
                    }
                }
            }
        }
    }
}

// ==================== Explicit Template Instantiations ====================

template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);