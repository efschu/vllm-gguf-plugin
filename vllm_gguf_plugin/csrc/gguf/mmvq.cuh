// copied and adapted from https://github.com/ggerganov/llama.cpp/blob/b2899/ggml-cuda/mmvq.cu
// copied and adapted from https://github.com/ggerganov/llama.cpp/blob/b2899/ggml-cuda/mmvq.cu
// Generalized to ncols_dst output columns per thread block (llama.cpp-style
// batched MMVQ): each quant block of W is loaded once per thread and dotted
// against up to ncols_dst activation vectors — the repeated vec_dot calls hit
// L1 for the weight bytes, so small decode batches (e.g. MTP: 1 + k drafts)
// run at nearly the cost of batch 1 instead of scaling linearly.
template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda, int ncols_dst>
static __global__ void mul_mat_vec_q(const void * __restrict__ vx, const void * __restrict__ vy, scalar_t * __restrict__ dst, const int ncols, const int nrows, const int nvecs) {
    const auto row = blockIdx.x*blockDim.y + threadIdx.y;
    const int vec0 = blockIdx.y * ncols_dst;

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    const int nrows_y = (ncols + 512 - 1) / 512 * 512;

    // partial sums for each thread, one per output column
    float tmp[ncols_dst];
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        tmp[j] = 0.0f;
    }

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (auto i = threadIdx.x / (qi/vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iqs  = vdr * (threadIdx.x % (qi/vdr)); // x block quant index when casting the quants to int

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            if (ncols_dst == 1 || vec0 + j < nvecs) {
                // y block index that aligns with ibx
                const int iby = (vec0 + j)*(nrows_y/QK8_1) + i * (qk/QK8_1);
                tmp[j] += vec_dot_q_cuda(&x[ibx], &y[iby], iqs);
            }
        }
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        float t = tmp[j];
#pragma unroll
        for (int mask = WARP_SIZE/2; mask > 0; mask >>= 1) {
            t += VLLM_SHFL_XOR_SYNC(t, mask);
        }
        if (threadIdx.x == 0 && (ncols_dst == 1 || vec0 + j < nvecs)) {
            dst[(vec0 + j)*nrows + row] = t;
        }
    }
}

// Launch helper: picks the widest ncols_dst that does not overshoot the
// batch too far. Grid y covers ceil(nvecs / ncols_dst) column groups.
template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
static void mmvq_launch(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    if (nvecs <= 1) {
        const dim3 block_nums(block_num_y, 1, 1);
        mul_mat_vec_q<scalar_t, qk, qi, block_q_t, vdr, vec_dot_q_cuda, 1>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
    } else if (nvecs <= 2) {
        const dim3 block_nums(block_num_y, 1, 1);
        mul_mat_vec_q<scalar_t, qk, qi, block_q_t, vdr, vec_dot_q_cuda, 2>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
    } else if (nvecs <= 4) {
        const dim3 block_nums(block_num_y, 1, 1);
        mul_mat_vec_q<scalar_t, qk, qi, block_q_t, vdr, vec_dot_q_cuda, 4>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
    } else {
        const dim3 block_nums(block_num_y, (nvecs + 7) / 8, 1);
        mul_mat_vec_q<scalar_t, qk, qi, block_q_t, vdr, vec_dot_q_cuda, 8>
            <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
    }
}

template<typename scalar_t>
static void mul_mat_vec_q4_0_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    mmvq_launch<scalar_t, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template<typename scalar_t>
static void mul_mat_vec_q4_1_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    mmvq_launch<scalar_t, QK4_0, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template<typename scalar_t>
static void mul_mat_vec_q5_0_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    mmvq_launch<scalar_t, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template<typename scalar_t>
static void mul_mat_vec_q5_1_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    mmvq_launch<scalar_t, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template<typename scalar_t>
static void mul_mat_vec_q8_0_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    mmvq_launch<scalar_t, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template<typename scalar_t>
static void mul_mat_vec_q2_K_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    mmvq_launch<scalar_t, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template<typename scalar_t>
static void mul_mat_vec_q3_K_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    mmvq_launch<scalar_t, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template<typename scalar_t>
static void mul_mat_vec_q4_K_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    mmvq_launch<scalar_t, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template<typename scalar_t>
static void mul_mat_vec_q5_K_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    mmvq_launch<scalar_t, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}


// Q6_K-specialized batched MMVQ: the generic path re-runs the full 6-bit
// unpack (shifts/masks/__vsubss4) once per output column, which makes it
// compute-bound and scale linearly with the batch. Here the per-thread
// weight loads and unpack are hoisted out of the column loop, so the
// marginal cost per extra column is two dp4a + the (L1-resident) q8_1
// activation loads — small decode batches stay near batch-1 cost.
template <typename scalar_t, int ncols_dst>
static __global__ void mul_mat_vec_q6_K_nc(const void * __restrict__ vx, const void * __restrict__ vy,
                                           scalar_t * __restrict__ dst, const int ncols, const int nrows, const int nvecs) {
    const auto row = blockIdx.x*blockDim.y + threadIdx.y;
    const int vec0 = blockIdx.y * ncols_dst;

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / QK_K;
    const int nrows_y = (ncols + 512 - 1) / 512 * 512;
    const int y_stride = nrows_y / QK8_1;

    float tmp[ncols_dst];
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        tmp[j] = 0.0f;
    }

    const block_q6_K * x = (const block_q6_K *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    const int iqs = threadIdx.x; // vdr=1, qi=QI6_K=32: one iqs slice per lane
    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));
    const int vh_idx = (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4);
    const int iqs8 = iqs % QI8_1;

    for (int i = 0; i < blocks_per_row; ++i) {
        const block_q6_K * bq6 = x + row*blocks_per_row + i;

        const int vl = get_int_from_uint8(bq6->ql, iqs);
        const int vh = get_int_from_uint8(bq6->qh, vh_idx) >> vh_shift;
        const int8_t * scales = bq6->scales + scale_offset;
        const float d = __half2float(bq6->d);

        int vi[QR6_K];
        int sc[QR6_K];
#pragma unroll
        for (int k = 0; k < QR6_K; ++k) {
            const int vil = (vl >> (4*k)) & 0x0F0F0F0F;
            const int vih = ((vh >> (4*k)) << 4) & 0x30303030;
            vi[k] = __vsubss4(vil | vih, 0x20202020);
            sc[k] = scales[4*k];
        }

        const int iby0 = i * (QK_K/QK8_1) + bq8_offset;

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            if (ncols_dst == 1 || vec0 + j < nvecs) {
                const block_q8_1 * bq8 = y + (vec0 + j)*y_stride + iby0;
                float sumf = 0.0f;
#pragma unroll
                for (int k = 0; k < QR6_K; ++k) {
                    const int u = get_int_from_int8_aligned(bq8[2*k].qs, iqs8);
                    const float d8 = __low2float(bq8[2*k].ds);
                    sumf += d8 * (__dp4a(vi[k], u, 0) * sc[k]);
                }
                tmp[j] += d * sumf;
            }
        }
    }

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        float t = tmp[j];
#pragma unroll
        for (int mask = WARP_SIZE/2; mask > 0; mask >>= 1) {
            t += VLLM_SHFL_XOR_SYNC(t, mask);
        }
        if (threadIdx.x == 0 && (ncols_dst == 1 || vec0 + j < nvecs)) {
            dst[(vec0 + j)*nrows + row] = t;
        }
    }
}

template<typename scalar_t>
static void mul_mat_vec_q6_K_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
    if (nvecs <= 1) {
        const dim3 block_nums(block_num_y, 1, 1);
        mul_mat_vec_q6_K_nc<scalar_t, 1><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
    } else if (nvecs <= 2) {
        const dim3 block_nums(block_num_y, 1, 1);
        mul_mat_vec_q6_K_nc<scalar_t, 2><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
    } else if (nvecs <= 4) {
        const dim3 block_nums(block_num_y, 1, 1);
        mul_mat_vec_q6_K_nc<scalar_t, 4><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
    } else {
        const dim3 block_nums(block_num_y, (nvecs + 7) / 8, 1);
        mul_mat_vec_q6_K_nc<scalar_t, 8><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
    }
}

template<typename scalar_t>
static void mul_mat_vec_iq2_xxs_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    mmvq_launch<scalar_t, QK_K, QI2_XXS, block_iq2_xxs, 1, vec_dot_iq2_xxs_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template<typename scalar_t>
static void mul_mat_vec_iq2_xs_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    mmvq_launch<scalar_t, QK_K, QI2_XS, block_iq2_xs, 1, vec_dot_iq2_xs_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template<typename scalar_t>
static void mul_mat_vec_iq2_s_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    mmvq_launch<scalar_t, QK_K, QI2_S, block_iq2_s, 1, vec_dot_iq2_s_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template<typename scalar_t>
static void mul_mat_vec_iq3_xxs_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    mmvq_launch<scalar_t, QK_K, QI3_XXS, block_iq3_xxs, 1, vec_dot_iq3_xxs_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template<typename scalar_t>
static void mul_mat_vec_iq1_s_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    mmvq_launch<scalar_t, QK_K, QI1_S, block_iq1_s, 1, vec_dot_iq1_s_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template<typename scalar_t>
static void mul_mat_vec_iq1_m_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    mmvq_launch<scalar_t, QK_K, QI1_M, block_iq1_m, 1, vec_dot_iq1_m_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template<typename scalar_t>
static void mul_mat_vec_iq4_nl_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    mmvq_launch<scalar_t, QK4_NL, QI4_NL, block_iq4_nl, VDR_Q4_0_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template<typename scalar_t>
static void mul_mat_vec_iq4_xs_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    mmvq_launch<scalar_t, QK_K, QI4_XS, block_iq4_xs, 1, vec_dot_iq4_xs_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template<typename scalar_t>
static void mul_mat_vec_iq3_s_q8_1_cuda(const void * vx, const void * vy, scalar_t * dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
    mmvq_launch<scalar_t, QK_K, QI3_XS, block_iq3_s, 1, vec_dot_iq3_s_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}
