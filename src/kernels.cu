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

#ifdef __ILUVATAR_FP16_H__

#else
// 否则提供简单的 half 类型定义
struct half {
    unsigned short data;
    
    half() : data(0) {}
    half(float f) {
        // 简单的 float 到 half 转换（非精确）
        unsigned int x = *reinterpret_cast<unsigned int*>(&f);
        unsigned int sign = (x >> 31) & 0x1;
        unsigned int exp = (x >> 23) & 0xFF;
        unsigned int mant = x & 0x7FFFFF;
        
        if (exp == 0xFF) { // NaN or Inf
            data = (sign << 15) | 0x7C00;
            if (mant) data |= 0x0200; // NaN
        } else if (exp == 0) { // Zero or denormal
            data = (sign << 15);
        } else {
            int new_exp = exp - 127 + 15;
            if (new_exp >= 31) { // Overflow
                data = (sign << 15) | 0x7C00;
            } else if (new_exp <= 0) { // Underflow
                data = (sign << 15);
            } else {
                data = (sign << 15) | (new_exp << 10) | (mant >> 13);
            }
        }
    }
    
    operator float() const {
        unsigned int sign = (data >> 15) & 0x1;
        unsigned int exp = (data >> 10) & 0x1F;
        unsigned int mant = data & 0x3FF;
        
        if (exp == 0x1F) { // NaN or Inf
            return (sign ? -1.0f : 1.0f) * (mant ? std::numeric_limits<float>::quiet_NaN() 
                                                : std::numeric_limits<float>::infinity());
        } else if (exp == 0) { // Zero or denormal
            return sign ? -0.0f : 0.0f;
        } else {
            float f = 1.0f + (mant / 1024.0f);
            f = f * powf(2.0f, static_cast<float>(exp) - 15.0f);
            return sign ? -f : f;
        }
    }
};
#endif

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
    h_o.resize(output_size, T(0));
    
    // Calculate scaling factor
    float scale_factor = 1.0f / sqrtf(static_cast<float>(head_dim));
    int head_repeat = query_heads / kv_heads;
    
    
    for (int b = 0; b < batch_size; ++b) {
        for (int tgt_i = 0; tgt_i < target_seq_len; ++tgt_i) {
            for (int qh = 0; qh < query_heads; ++qh) {
                // GQA mapping
                int kvh = qh / head_repeat;
                if (kvh >= kv_heads) kvh = kv_heads - 1;
                
                // Compute attention scores
                std::vector<float> scores(src_seq_len, 0.0f);
                
                for (int src_j = 0; src_j < src_seq_len; ++src_j) {
                    // Apply causal mask
                    if (is_causal && tgt_i < src_j) {
                        scores[src_j] = -1e20f;
                        continue;
                    }
                    
                    // Compute dot product
                    float dot_product = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        size_t q_idx = ((b * target_seq_len + tgt_i) * query_heads + qh) * head_dim + d;
                        size_t k_idx = ((b * src_seq_len + src_j) * kv_heads + kvh) * head_dim + d;
                        
                        if (std::is_same<T, float>::value) {
                            dot_product += h_q[q_idx] * h_k[k_idx];
                        } else {
                            // For half or other types, convert to float
                            float q_val, k_val;
                            if (std::is_same<T, half>::value) {
                                q_val = static_cast<float>(h_q[q_idx]);
                                k_val = static_cast<float>(h_k[k_idx]);
                            } else {
                                q_val = static_cast<float>(h_q[q_idx]);
                                k_val = static_cast<float>(h_k[k_idx]);
                            }
                            dot_product += q_val * k_val;
                        }
                    }
                    
                    scores[src_j] = dot_product * scale_factor;
                }
                
                // Find max for numerical stability
                float max_score = scores[0];
                for (int src_j = 1; src_j < src_seq_len; ++src_j) {
                    if (scores[src_j] > max_score) {
                        max_score = scores[src_j];
                    }
                }
                
                // Compute softmax
                float exp_sum = 0.0f;
                std::vector<float> exp_scores(src_seq_len, 0.0f);
                
                for (int src_j = 0; src_j < src_seq_len; ++src_j) {
                    if (scores[src_j] < -1e19f) {
                        exp_scores[src_j] = 0.0f;
                    } else {
                        exp_scores[src_j] = expf(scores[src_j] - max_score);
                        exp_sum += exp_scores[src_j];
                    }
                }
                
                // Compute output
                for (int d = 0; d < head_dim; ++d) {
                    float weighted_sum = 0.0f;
                    
                    if (exp_sum > 1e-12f) {
                        float inv_exp_sum = 1.0f / exp_sum;
                        
                        for (int src_j = 0; src_j < src_seq_len; ++src_j) {
                            if (exp_scores[src_j] > 0.0f) {
                                size_t v_idx = ((b * src_seq_len + src_j) * kv_heads + kvh) * head_dim + d;
                                float v_val;
                                
                                if (std::is_same<T, float>::value) {
                                    v_val = h_v[v_idx];
                                } else {
                                    v_val = static_cast<float>(h_v[v_idx]);
                                }
                                
                                weighted_sum += (exp_scores[src_j] * inv_exp_sum) * v_val;
                            }
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