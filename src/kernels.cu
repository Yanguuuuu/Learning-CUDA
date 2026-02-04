#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */

// 树形规约
template<typename T>
__global__ void trace_kernel(T *output,T const * d_input ,size_t n)
{ 

  extern __shared__ __align__(sizeof(T)) unsigned char shared_mem[];
  T * smem = reinterpret_cast<T*>(shared_mem);
  //extern __shared__ float smem[];  // shared_memory在block内共享
  size_t thread_id = threadIdx.x;
  size_t idx = blockIdx.x * blockDim.x + thread_id;  // 每个thread分配一个idx数据

  smem[thread_id] = (idx < n)? d_input[idx]:0;  // 数据复制到shared_memory
  __syncthreads();

  for(int s = blockDim.x / 2 ; s > 0; s >>= 1)     // 规约合并--二分合并，
    {
        if(thread_id < s)
        {
            smem[thread_id] += smem[thread_id + s];  // blockdim = 8时 ;block内部进行规约 1-8 变成   1-4，（1，5）-> 1，（2，6）
        }
        __syncthreads();
    }

    if(thread_id == 0)    // block间规约
    {
        atomicAdd(output,smem[0]);
    }

}

template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  // TODO: Implement the trace function
  /* 1.cpu提取对角线元素置于gpu规约，减小拷贝开销 */ 
  size_t size = std::min(rows,cols);        // 拷贝的数据数量
  size_t size_bytes = size * sizeof(T);     // 需要分配的字节数量
  std::vector<T>  h_dim(size);
  for(size_t i = 0 ; i < size ;++i)
  {
      h_dim[i] = h_input[i * cols + i];
  }

  T *h_res = new T(0);  // host侧结果指针
  // cuda侧分配内存
  T *d_res;  // device侧指针
  T *d_dim;  // device侧用于传输结果

  cudaMalloc(&d_res,sizeof(T));
  cudaMalloc(&d_dim,size_bytes);

  // host2device 数据迁移至device侧
  cudaMemcpy(d_dim,h_dim.data(),size_bytes,cudaMemcpyHostToDevice);

  /* 2.设置核函数超参数,对角线元素规约 */
  dim3 block_dim(256);
  dim3 grid_dim((size + block_dim.x - 1) / block_dim.x);
  trace_kernel<<<grid_dim,block_dim>>>(d_res,d_dim,size); 

  // device2host 数据迁移至host侧
  cudaMemcpy(h_res,d_res,sizeof(T),cudaMemcpyDeviceToHost);

  // 数据清理
  cudaFree(d_res);
  cudaFree(d_dim);
  
  return *h_res;
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */


template<typename T>
__global__ void flash_attention_kernel(
    T* output,
    const T* query,
    const T* key,
    const T* value,
    int batch_size,
    int tgt_seq_len,
    int src_seq_len,
    int query_heads,
    int kv_heads,
    int head_dim,
    float scale,
    bool is_causal) {
    
    // 每个线程处理一个输出位置
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y; 
    int tgt_pos = blockIdx.x * blockDim.x + threadIdx.x;
    
    
    if (batch_idx >= batch_size || head_idx >= query_heads || tgt_pos >= tgt_seq_len) {
        return;
    }
    
    // 获取当前query的指针 query_stride分别为{tgt_seq_len * query_heads * head_dim,query_heads * head_dim,head_dim,1}
    const T* q_ptr = query + batch_idx * tgt_seq_len * query_heads * head_dim +tgt_pos * query_heads * head_dim +head_idx * head_dim;
    
    // GQA 实现广播映射: queryheadidx 映射为 kv_head_idx
    int kv_head_idx = head_idx * kv_heads / query_heads;
    
    // 分配本地内存存储中间结果
    float* local_q = new float[head_dim];
    float* local_out = new float[head_dim];
    float max_val = -1e30f;
    float sum_exp = 0.0f;
    
    // 加载query到本地内存
    for (int i = 0; i < head_dim; i++) {
        local_q[i] = static_cast<float>(q_ptr[i]);
        local_out[i] = 0.0f;
    }
    
    // 计算attention scores
    for (int src_pos = 0; src_pos < src_seq_len; src_pos++) {
        // 检查causal masking
        if (is_causal && src_pos > tgt_pos) {
            continue;
        }
        
        // 获取当前key/value
        const T* k_ptr = key +  batch_idx * src_seq_len * kv_heads * head_dim +src_pos * kv_heads * head_dim + kv_head_idx * head_dim;
        
        const T* v_ptr = value + batch_idx * src_seq_len * kv_heads * head_dim +src_pos * kv_heads * head_dim +kv_head_idx * head_dim;
        
        // 计算Q·K^T
        float dot_product = 0.0f;
        for (int i = 0; i < head_dim; i++) {
            float k_val = static_cast<float>(k_ptr[i]);
            dot_product += local_q[i] * k_val;
        }
        
        float score = dot_product * scale;
        
        // 更新最大值（第一遍找到最大值用于数值稳定）
        if (score > max_val) {
            max_val = score;
        }
    }
    
    // 计算softmax和加权求和
    for (int src_pos = 0; src_pos < src_seq_len; src_pos++) {
        if (is_causal && src_pos > tgt_pos) {
            continue;
        }
        
        const T* k_ptr = key + batch_idx * src_seq_len * kv_heads * head_dim + src_pos * kv_heads * head_dim +kv_head_idx * head_dim;
        
        const T* v_ptr = value + batch_idx * src_seq_len * kv_heads * head_dim +src_pos * kv_heads * head_dim +kv_head_idx * head_dim;
        
        // 重新计算score
        float dot_product = 0.0f;
        for (int i = 0; i < head_dim; i++) {
            float k_val = static_cast<float>(k_ptr[i]);
            dot_product += local_q[i] * k_val;
        }
        
        float score = dot_product * scale;
        
        // 计算计算 safe-softmax 
        float exp_val = expf(score - max_val);
        sum_exp += exp_val;
        
        // exp(Q @ K ^T - max) / sqrt(head_dim) * V  后续需要进行除上指数和
        for (int i = 0; i < head_dim; i++) {
            float v_val = static_cast<float>(v_ptr[i]);
            local_out[i] += exp_val * v_val;
        }
    }
    
    // 写入输出
    T* out_ptr = output + batch_idx * tgt_seq_len * query_heads * head_dim +tgt_pos * query_heads * head_dim +head_idx * head_dim;
    
    if (sum_exp > 0.0f) {
        float inv_sum = 1.0f / sum_exp;
        for (int i = 0; i < head_dim; i++) {
            out_ptr[i] = static_cast<T>(local_out[i] * inv_sum);
        }
    } else {
        for (int i = 0; i < head_dim; i++) {
            out_ptr[i] = static_cast<T>(0.0f);
        }
    }
    
    delete[] local_q;
    delete[] local_out;
}



template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
  // TODO: Implement the flash attention function
    // 分配输出内存
  size_t total_q_size = (size_t)batch_size * target_seq_len * query_heads * head_dim;
  size_t total_kv_size = (size_t)batch_size * src_seq_len * kv_heads * head_dim;
  h_o.resize(total_q_size);
  
  // 分配设备内存
  T* d_q;
  T* d_k;
  T* d_v;
  T* d_o;
  
  size_t q_bytes = total_q_size * sizeof(T);
  size_t kv_bytes = total_kv_size * sizeof(T);
  
  cudaMalloc(&d_q, q_bytes);
  cudaMalloc(&d_k, kv_bytes);
  cudaMalloc(&d_v, kv_bytes);
  cudaMalloc(&d_o, q_bytes);
  
  // 拷贝数据到设备
  cudaMemcpy(d_q, h_q.data(), q_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, h_k.data(), kv_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v.data(), kv_bytes, cudaMemcpyHostToDevice);
  
  // 计算缩放因子
  float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  
  // 简单的核函数配置：每个线程处理一个输出位置
  dim3 block_dim(128);  // 每个block 128个线程
  dim3 grid_dim((target_seq_len + block_dim.x - 1) / block_dim.x,
                query_heads,
                batch_size);
  
  // 启动kernel
  flash_attention_kernel<T><<<grid_dim, block_dim>>>(
      d_o, d_q, d_k, d_v,
      batch_size, target_seq_len, src_seq_len,
      query_heads, kv_heads, head_dim,
      scale, is_causal);

  // 拷贝结果回host
  cudaMemcpy(h_o.data(), d_o, q_bytes, cudaMemcpyDeviceToHost);
  
  // 清理设备内存
  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_o);


}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
