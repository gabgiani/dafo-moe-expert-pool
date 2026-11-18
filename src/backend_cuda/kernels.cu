// Kernels of the CUDA backend: the same kernels, with the same parameters in the same order, as backend_metal/kernels.metal and
// matmul.metal, written in CUDA C++ (compiled at run time by NVRTC). The compiler writes a header with the constants of the model before
// this text:   NR, NSG   rows per warp and warps per block of the matrix-vector products; STATE_VISION, ROUTE_MAX_WIDTH, ROUTE_MAX_EXPERTS,
// ROUTE_MAX_USED. A Metal "SIMD group" is a warp, a "thread group" a block. Activations are F32; keys and values are half.
//
// state[0] = position of the first token, state[2] = token chosen by the head, state[3] = tokens of the batch that are real,
// state[8 + t] = id of the t-th token, state[STATE_VISION + t] = last position the t-th token may attend to when it belongs to an image.
// No headers: NVRTC compiles this text on its own (half conversions are written with PTX).
#define INFINITY __int_as_float(0x7f800000)

typedef unsigned int uint;
typedef unsigned long long ulong;
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef long long int64_t;
typedef unsigned long long uint64_t;
typedef int int32_t;
typedef signed char int8_t;

#define KERNEL extern "C" __global__

__device__ __forceinline__ float simd_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float simd_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ uint simd_min_u(uint v) {
    for (int o = 16; o > 0; o >>= 1) v = min(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float h2f(ushort h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}
__device__ __forceinline__ ushort f2h(float f) {
    ushort h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}
__device__ __forceinline__ float load_half(const uchar *p) { return h2f(*(const ushort *)p); }

// Experts live in two places: slots below `dev_slots` in device memory (`dev`), the rest in pinned host memory (`host`, read by the GPU
// over PCIe). A model whose experts all fit in the device passes dev_slots larger than any slot.
__device__ __forceinline__ const uchar *expert_at(const uchar *dev, const uchar *host, uint dev_slots, uint slot, ulong stride) {
    return slot < dev_slots ? dev + (ulong)slot * stride : host + (ulong)(slot - dev_slots) * stride;
}

struct ushort2_t {
    ushort x, y;
};

struct block_q8_0 {
    ushort d;
    int8_t qs[32];
};

// ---- embedding ----------------------------------------------------------------------------------------------------

KERNEL void embed_q8(const uchar *table, const uint *state, float *out, uint width, float scale, const float *extra) {
    const uint i = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width) return;
    const uint token = state[8 + y];
    if (token & 0x80000000u) {
        out[y * width + i] = extra[(ulong)(token & 0x7fffffffu) * width + i];
        return;
    }
    const ulong row_bytes = (ulong)(width / 32) * 34;
    const uchar *block = table + (ulong)token * row_bytes + (ulong)(i / 32) * 34;
    out[y * width + i] = float(((const int8_t *)(block + 2))[i % 32]) * load_half(block) * scale;
}

KERNEL void zero_u32(uint *buffer, uint n) {
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buffer[i] = 0;
}

// ---- normalization ------------------------------------------------------------------------------------------------

// One block per group of `width` values; `has_scale` selects the learned scale.
KERNEL void rms_norm(const float *x, const float *w, float *y, uint width, float eps, uint has_scale) {
    __shared__ float partial[32];
    const uint group = blockIdx.x, t = threadIdx.x, threads = blockDim.x;
    x += (ulong)group * width;
    y += (ulong)group * width;
    float sum = 0;
    for (uint i = t; i < width; i += threads) sum += x[i] * x[i];
    sum = simd_sum(sum);
    if (t % 32 == 0) partial[t / 32] = sum;
    __syncthreads();
    if (t < 32) {
        float value = t < threads / 32 ? partial[t] : 0.0f;
        value = simd_sum(value);
        if (t == 0) partial[0] = value;
    }
    __syncthreads();
    const float inverse = rsqrtf(partial[0] / float(width) + eps);
    for (uint i = t; i < width; i += threads) y[i] = x[i] * inverse * (has_scale ? w[i] : 1.0f);
}

KERNEL void scale_by(const float *x, const float *w, float *y, uint n) {
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * w[0];
}

KERNEL void add(const float *a, const float *b, float *y, uint n) {
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}

__device__ __forceinline__ float gelu_tanh(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

KERNEL void gelu_mul(const float *gate, const float *up, float *out, uint n, uint gate_stride, uint up_stride, uint out_stride) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= n) return;
    out[y * out_stride + x] = gelu_tanh(gate[y * gate_stride + x]) * up[y * up_stride + x];
}

// ---- matrix times vector ------------------------------------------------------------------------------------------

KERNEL void mv_q8_0(const block_q8_0 *x, const float *y, float *dst, uint width, uint rows) {
    const uint lane = threadIdx.x & 31, simd = threadIdx.x >> 5;
    const uint nb = width / 32;
    const uint first = (blockIdx.x * NSG + simd) * NR;
    if (first >= rows) return;
    const block_q8_0 *base = x + (ulong)first * nb;
    float sums[NR];
    for (int r = 0; r < NR; ++r) sums[r] = 0;
    const uint ix = lane / 4, il = lane % 4;
    const float *yb = y + ix * 32 + 8 * il;
    float yl[8];
    for (uint ib = ix; ib < nb; ib += 8) {
        for (int i = 0; i < 8; ++i) yl[i] = yb[i];
        for (int r = 0; r < NR; ++r) {
            const block_q8_0 *block = base + (ulong)r * nb + ib;
            const int8_t *qs = block->qs + 8 * il;
            float sum = 0;
            for (int i = 0; i < 8; ++i) sum += qs[i] * yl[i];
            sums[r] += sum * h2f(block->d);
        }
        yb += 8 * 32;
    }
    for (int r = 0; r < NR; ++r) {
        const float total = simd_sum(sums[r]);
        if (lane == 0 && first + r < rows) dst[first + r] = total;
    }
}

__device__ __forceinline__ void scale_min_k4(uint j, const uchar *q, float &scale, float &minimum) {
    if (j < 4) {
        scale = float(q[j] & 63);
        minimum = float(q[j + 4] & 63);
    } else {
        scale = float((q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4));
        minimum = float((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

// Expert matrices in Q4_K (144-byte blocks of 256 values); see the Metal kernel of the same name for the layout.
KERNEL void mv_id_q4_k(const uchar *w, const uint *ids, const float *x, float *y, uint width, uint rows, uint x_stride, const uchar *w_host, uint dev_slots) {
    const uint lane = threadIdx.x & 31, simd = threadIdx.x >> 5;
    const uint slot = blockIdx.y;
    const uint nb = width / 256;
    const ulong row_bytes = (ulong)nb * 144;
    const uchar *base = expert_at(w, w_host, dev_slots, ids[slot], (ulong)rows * row_bytes);
    const uint first = (blockIdx.x * NSG + simd) * NR;
    if (first >= rows) return;
    const float *xv = x + slot * x_stride;
    float sums[NR];
    for (int r = 0; r < NR; ++r) sums[r] = 0;
    const uint ix = lane / 8, it = lane % 8, pair = it / 2, half_index = it % 2;
    for (uint ib = ix; ib < nb; ib += 4) {
        const uint start = ib * 256 + pair * 64 + half_index * 16;
        float yl[16], yh[16];
        float sum_low = 0, sum_high = 0;
        for (int i = 0; i < 16; ++i) {
            yl[i] = xv[start + i];
            yh[i] = xv[start + 32 + i];
            sum_low += yl[i];
            sum_high += yh[i];
        }
        for (int r = 0; r < NR; ++r) {
            const uchar *block = base + (ulong)(first + r) * row_bytes + (ulong)ib * 144;
            const float d = load_half(block), dmin = load_half(block + 2);
            float scale_low, min_low, scale_high, min_high;
            scale_min_k4(2 * pair, block + 4, scale_low, min_low);
            scale_min_k4(2 * pair + 1, block + 4, scale_high, min_high);
            const uint *packed = (const uint *)(block + 16 + pair * 32 + half_index * 16);
            float acc_low = 0, acc_high = 0;
            for (int word = 0; word < 4; ++word) {
                const uint value = packed[word];
                for (int k = 0; k < 4; ++k) {
                    const uint byte = (value >> (8 * k)) & 0xff;
                    acc_low += float(byte & 0x0f) * yl[word * 4 + k];
                    acc_high += float(byte >> 4) * yh[word * 4 + k];
                }
            }
            sums[r] += d * (scale_low * acc_low + scale_high * acc_high) - dmin * (min_low * sum_low + min_high * sum_high);
        }
    }
    for (int r = 0; r < NR; ++r) {
        const float total = simd_sum(sums[r]);
        if (lane == 0 && first + r < rows) y[slot * rows + first + r] = total;
    }
}

// Expert matrices in Q5_1 (24-byte blocks of 32 values: half d, half m, u32 high bits, 16 bytes of low nibbles).
KERNEL void mv_id_q5_1(const uchar *w, const uint *ids, const float *x, float *y, uint width, uint rows, uint x_stride, const uchar *w_host, uint dev_slots) {
    const uint lane = threadIdx.x & 31, simd = threadIdx.x >> 5;
    const uint slot = blockIdx.y;
    const uint nb = width / 32;
    const ulong row_bytes = (ulong)nb * 24;
    const uchar *base = expert_at(w, w_host, dev_slots, ids[slot], (ulong)rows * row_bytes);
    const uint first = (blockIdx.x * NSG + simd) * NR;
    if (first >= rows) return;
    const float *xv = x + slot * x_stride;
    float sums[NR];
    for (int r = 0; r < NR; ++r) sums[r] = 0;
    const uint ix = lane / 2, h = lane % 2;
    for (uint ib = ix; ib < nb; ib += 16) {
        float yl[8], yh[8];
        float sum_x = 0;
        for (int i = 0; i < 8; ++i) {
            yl[i] = xv[ib * 32 + h * 8 + i];
            yh[i] = xv[ib * 32 + 16 + h * 8 + i];
            sum_x += yl[i] + yh[i];
        }
        for (int r = 0; r < NR; ++r) {
            const uchar *block = base + (ulong)(first + r) * row_bytes + (ulong)ib * 24;
            const float d = load_half(block), m = load_half(block + 2);
            const uint high = *(const uint *)(block + 4);
            const uint *packed = (const uint *)(block + 8 + h * 8);
            float acc = 0;
            for (int word = 0; word < 2; ++word) {
                const uint value = packed[word];
                for (int k = 0; k < 4; ++k) {
                    const uint j = h * 8 + word * 4 + k;
                    const uint byte = (value >> (8 * k)) & 0xff;
                    const uint low_q = (byte & 0x0f) | (((high >> j) << 4) & 0x10);
                    const uint high_q = (byte >> 4) | ((high >> (j + 12)) & 0x10);
                    acc += float(low_q) * yl[word * 4 + k] + float(high_q) * yh[word * 4 + k];
                }
            }
            sums[r] += d * acc + m * sum_x;
        }
    }
    for (int r = 0; r < NR; ++r) {
        const float total = simd_sum(sums[r]);
        if (lane == 0 && first + r < rows) y[slot * rows + first + r] = total;
    }
}

KERNEL void mv_id_q8_0(const uchar *w, const uint *ids, const float *x, float *y, uint width, uint rows, uint x_stride, const uchar *w_host, uint dev_slots) {
    const uint lane = threadIdx.x & 31, simd = threadIdx.x >> 5;
    const uint slot = blockIdx.y;
    const uint nb = width / 32;
    const uint first = (blockIdx.x * NSG + simd) * NR;
    if (first >= rows) return;
    const block_q8_0 *base = (const block_q8_0 *)expert_at(w, w_host, dev_slots, ids[slot], (ulong)rows * nb * 34) + (ulong)first * nb;
    const float *xv = x + slot * x_stride;
    float sums[NR];
    for (int r = 0; r < NR; ++r) sums[r] = 0;
    const uint ix = lane / 4, il = lane % 4;
    float yl[8];
    for (uint ib = ix; ib < nb; ib += 8) {
        for (int i = 0; i < 8; ++i) yl[i] = xv[ib * 32 + 8 * il + i];
        for (int r = 0; r < NR; ++r) {
            const block_q8_0 *block = base + (ulong)r * nb + ib;
            const int8_t *qs = block->qs + 8 * il;
            float sum = 0;
            for (int i = 0; i < 8; ++i) sum += qs[i] * yl[i];
            sums[r] += sum * h2f(block->d);
        }
    }
    for (int r = 0; r < NR; ++r) {
        const float total = simd_sum(sums[r]);
        if (lane == 0 && first + r < rows) y[slot * rows + first + r] = total;
    }
}

KERNEL void experts_sum(const float *in, const float *weights, float *out, uint width, uint slots) {
    const uint i = blockIdx.x * blockDim.x + threadIdx.x, token = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width) return;
    float sum = 0;
    for (uint s = 0; s < slots; ++s) sum += weights[token * slots + s] * in[(token * slots + s) * width + i];
    out[token * width + i] = sum;
}

// ---- router -------------------------------------------------------------------------------------------------------

// One block of 1024 threads per token: normalizes the input, scales it, computes the router logits, blocks the experts the host has not
// loaded, picks the `used` largest (up to 128 experts), writes the pool slot and the weight of each and counts what the router wanted.
KERNEL void route(const float *x, const float *router_scale, const float *matrix, const float *expert_scale, uint *ids, float *weights,
                  uint width, uint total, uint used, float eps, const uint *state, uint mask_at, uint *demand, uint demand_at) {
    const uint token = blockIdx.x, t = threadIdx.x, lane = t & 31, simd = t >> 5;
    x += (ulong)token * width;
    ids += token * used;
    weights += token * used;
    __shared__ float normalized[ROUTE_MAX_WIDTH];
    __shared__ float logits[ROUTE_MAX_EXPERTS];
    __shared__ float raw[ROUTE_MAX_EXPERTS];
    __shared__ float partial[32];
    float sum = 0;
    for (uint i = t; i < width; i += 1024) sum += x[i] * x[i];
    sum = simd_sum(sum);
    if (lane == 0) partial[simd] = sum;
    __syncthreads();
    if (t < 32) {
        const float value = simd_sum(partial[t]);
        if (t == 0) partial[0] = value;
    }
    __syncthreads();
    const float factor = rsqrtf(partial[0] / float(width) + eps) / sqrtf(float(width));
    for (uint i = t; i < width; i += 1024) normalized[i] = x[i] * factor * router_scale[i];
    __syncthreads();
    for (uint row = simd; row < total; row += 32) {
        float dot = 0;
        for (uint i = lane; i < width; i += 32) dot += normalized[i] * matrix[(ulong)row * width + i];
        dot = simd_sum(dot);
        if (lane == 0) {
            raw[row] = dot;
            logits[row] = state[mask_at + row] == 0xffffffffu ? -INFINITY : dot;
        }
    }
    __syncthreads();
    if (simd == 0) {
        constexpr uint K = (ROUTE_MAX_EXPERTS + 31) / 32;
        float held[K];
        for (uint k = 0; k < K; ++k) held[k] = lane + 32 * k < total ? logits[lane + 32 * k] : -INFINITY;
        uint chosen[ROUTE_MAX_USED];
        float values[ROUTE_MAX_USED];
        for (uint s = 0; s < used; ++s) {
            float mine = held[0];
            uint slot = 0;
            for (uint k = 1; k < K; ++k) {
                if (held[k] > mine) { mine = held[k]; slot = k; }
            }
            const uint index = lane + 32 * slot;
            const float best = simd_max(mine);
            const uint winner = simd_min_u(mine == best ? index : 0xffffffffu);
            chosen[s] = winner;
            values[s] = best;
            if ((winner & 31u) == lane) {
                for (uint k = 0; k < K; ++k) {
                    if (k == (winner >> 5)) held[k] = -INFINITY;
                }
            }
        }
        float wanted[K];
        for (uint k = 0; k < K; ++k) wanted[k] = lane + 32 * k < total ? raw[lane + 32 * k] : -INFINITY;
        for (uint s = 0; s < used; ++s) {
            float mine = wanted[0];
            uint slot = 0;
            for (uint k = 1; k < K; ++k) {
                if (wanted[k] > mine) { mine = wanted[k]; slot = k; }
            }
            const uint index = lane + 32 * slot;
            const float best = simd_max(mine);
            const uint winner = simd_min_u(mine == best ? index : 0xffffffffu);
            if (lane == 0) atomicAdd(&demand[demand_at + winner], 1u);
            if ((winner & 31u) == lane) {
                for (uint k = 0; k < K; ++k) {
                    if (k == (winner >> 5)) wanted[k] = -INFINITY;
                }
            }
        }
        if (lane == 0) {
            float largest = values[0], denominator = 0;
            for (uint s = 0; s < used; ++s) {
                values[s] = expf(values[s] - largest);
                denominator += values[s];
            }
            for (uint s = 0; s < used; ++s) {
                const uint word = state[mask_at + chosen[s]];
                ids[s] = word == 0 ? chosen[s] : word - 1;
                weights[s] = values[s] / denominator * expert_scale[chosen[s]];
            }
        }
    }
}

// ---- rotary embedding and cache -----------------------------------------------------------------------------------

KERNEL void rope(const float *in, float *out, const float *inverse, const uint *state, uint head_dim, uint heads) {
    const uint px = blockIdx.x * blockDim.x + threadIdx.x, py = blockIdx.y * blockDim.y + threadIdx.y, pz = blockIdx.z * blockDim.z + threadIdx.z;
    const uint half_dim = head_dim / 2;
    if (px >= half_dim || py >= heads) return;
    const float angle = float(state[0] + pz) * inverse[px];
    const float c = cosf(angle), s = sinf(angle);
    const uint base = (pz * heads + py) * head_dim;
    const float x1 = in[base + px], x2 = in[base + px + half_dim];
    out[base + px] = x1 * c - x2 * s;
    out[base + px + half_dim] = x2 * c + x1 * s;
}

KERNEL void cache_write(const float *key, const float *value, ushort *key_cache, ushort *value_cache, const uint *state, uint head_dim,
                        uint capacity, uint kv_heads) {
    const uint px = blockIdx.x * blockDim.x + threadIdx.x, py = blockIdx.y * blockDim.y + threadIdx.y, pz = blockIdx.z * blockDim.z + threadIdx.z;
    if (px >= head_dim || pz >= state[3]) return;
    const uint slot = (state[0] + pz) % capacity;
    const ulong destination = ((ulong)py * capacity + slot) * head_dim + px;
    const ulong source = ((ulong)pz * kv_heads + py) * head_dim + px;
    key_cache[destination] = f2h(key[source]);
    value_cache[destination] = f2h(value[source]);
}

// ---- attention ----------------------------------------------------------------------------------------------------

// One block (8 warps) per query head and token. Each warp walks its share of the positions with an online softmax; the warps are merged.
template <uint HEAD_DIM>
__device__ __forceinline__ void attention_impl(const float *query, const ushort *key_cache, const ushort *value_cache, float *out, const uint *state,
                                               uint heads, uint kv_heads, uint capacity, uint window) {
    constexpr uint PER_LANE = HEAD_DIM / 32;
    __shared__ float merged[8 * HEAD_DIM];
    __shared__ float maxima[8], sums[8];
    const uint lane = threadIdx.x & 31, simd = threadIdx.x >> 5;
    const uint head = blockIdx.x;
    const uint position = state[0] + blockIdx.y;
    uint last = position;
    if (window > 0) {
        const uint image_end = state[STATE_VISION + blockIdx.y];
        if (image_end > position) last = min(image_end, position + window - 1);
    }
    const uint begin = window > 0 && position + 1 > window ? position + 1 - window : 0;
    const ulong base = (ulong)(head / (heads / kv_heads)) * capacity * HEAD_DIM;
    float q[PER_LANE], accumulator[PER_LANE];
    for (uint j = 0; j < PER_LANE; ++j) {
        q[j] = query[(blockIdx.y * heads + head) * HEAD_DIM + lane + 32 * j];
        accumulator[j] = 0;
    }
    float maximum = -1e30f, sum = 0;
    for (uint p = begin + simd; p <= last; p += 8) {
        const ulong offset = base + (ulong)(p % capacity) * HEAD_DIM;
        float dot = 0;
        for (uint j = 0; j < PER_LANE; ++j) dot += q[j] * h2f(key_cache[offset + lane + 32 * j]);
        dot = simd_sum(dot);
        const float next = fmaxf(maximum, dot);
        const float shrink = expf(maximum - next), weight = expf(dot - next);
        sum = sum * shrink + weight;
        for (uint j = 0; j < PER_LANE; ++j) accumulator[j] = accumulator[j] * shrink + weight * h2f(value_cache[offset + lane + 32 * j]);
        maximum = next;
    }
    for (uint j = 0; j < PER_LANE; ++j) merged[simd * HEAD_DIM + lane + 32 * j] = accumulator[j];
    if (lane == 0) { maxima[simd] = maximum; sums[simd] = sum; }
    __syncthreads();
    if (simd == 0) {
        float overall = -1e30f;
        for (uint s = 0; s < 8; ++s) overall = fmaxf(overall, maxima[s]);
        float denominator = 0;
        float factors[8];
        for (uint s = 0; s < 8; ++s) {
            factors[s] = expf(maxima[s] - overall);
            denominator += factors[s] * sums[s];
        }
        for (uint j = 0; j < PER_LANE; ++j) {
            float value = 0;
            for (uint s = 0; s < 8; ++s) value += factors[s] * merged[s * HEAD_DIM + lane + 32 * j];
            out[(blockIdx.y * heads + head) * HEAD_DIM + lane + 32 * j] = value / denominator;
        }
    }
}

KERNEL void attention_256(const float *query, const ushort *key_cache, const ushort *value_cache, float *out, const uint *state, uint heads,
                          uint kv_heads, uint capacity, uint window) {
    attention_impl<256>(query, key_cache, value_cache, out, state, heads, kv_heads, capacity, window);
}
KERNEL void attention_512(const float *query, const ushort *key_cache, const ushort *value_cache, float *out, const uint *state, uint heads,
                          uint kv_heads, uint capacity, uint window) {
    attention_impl<512>(query, key_cache, value_cache, out, state, heads, kv_heads, capacity, window);
}

// ---- head ---------------------------------------------------------------------------------------------------------

// The token with the highest logit (the lowest id on a tie): one block of 1024 threads.
KERNEL void argmax_token(const float *logits, uint *state, uint n) {
    __shared__ float values[32];
    __shared__ uint indices[32];
    const uint t = threadIdx.x, lane = t & 31, simd = t >> 5;
    float best = -INFINITY;
    uint best_index = 0;
    for (uint i = t; i < n; i += 1024) {
        if (logits[i] > best) {
            best = logits[i];
            best_index = i;
        }
    }
    for (uint offset = 16; offset > 0; offset /= 2) {
        const float other = __shfl_down_sync(0xffffffffu, best, offset);
        const uint other_index = __shfl_down_sync(0xffffffffu, best_index, offset);
        if (other > best || (other == best && other_index < best_index)) {
            best = other;
            best_index = other_index;
        }
    }
    if (lane == 0) {
        values[simd] = best;
        indices[simd] = best_index;
    }
    __syncthreads();
    if (t == 0) {
        float winner = values[0];
        uint winner_index = indices[0];
        for (uint s = 1; s < 32; ++s) {
            if (values[s] > winner || (values[s] == winner && indices[s] < winner_index)) {
                winner = values[s];
                winner_index = indices[s];
            }
        }
        state[2] = winner_index;
    }
}

// ---- experts in host memory ---------------------------------------------------------------------------------------

struct __align__(16) u4 {
    uint x, y, z, w;
};

// Copies the experts of `ids[0..used]` that live in host memory into `staging` (expert slot s goes to staging + s * stride), with 16-byte
// accesses so the PCIe bus is read in full lines. Grid: x = 16-byte pieces of an expert (256 per block), y = slot.
KERNEL void stage_experts(const uint *ids, const uchar *host, uchar *staging, uint dev_slots, uint stride, uint used) {
    const uint piece = blockIdx.x * blockDim.x + threadIdx.x;
    const uint s = blockIdx.y;
    if (s >= used || piece * 16 >= stride) return;
    const uint expert = ids[s];
    if (expert < dev_slots) return;
    const u4 *source = (const u4 *)(host + (ulong)(expert - dev_slots) * stride);
    u4 *destination = (u4 *)(staging + (ulong)s * stride);
    destination[piece] = source[piece];
}

// Copies the whole host part of one layer's expert tensor (`count` experts) into `staging`, for the products that read every expert (prompt reading).
KERNEL void stage_all(const uchar *host, uchar *staging, ulong bytes) {
    const ulong piece = (ulong)blockIdx.x * blockDim.x + threadIdx.x;
    if (piece * 16 >= bytes) return;
    ((u4 *)staging)[piece] = ((const u4 *)host)[piece];
}

// ---- matrix times matrix (prompt reading) --------------------------------------------------------------------------

// A tile product: 64 rows of a quantized matrix times 32 vectors, 32 values of K at a time, dequantizing on the fly and accumulating in F32.
// 128 threads; thread t loads 16 values of one weight row (row t / 2, half t % 2) and 8 values of one vector (vector t / 4), and
// computes 4 rows x 4 vectors.
#define ENC_Q8_0 0
#define ENC_Q5_1 1
#define ENC_Q4_K 2

template <int ENC>
__device__ __forceinline__ void load_weights16(const uchar *row, uint k0, uint half_index, float *out) {
    if (ENC == ENC_Q8_0) {
        const block_q8_0 *block = (const block_q8_0 *)row + (k0 / 32);
        const float d = h2f(block->d);
        for (int i = 0; i < 16; ++i) out[i] = d * float(block->qs[half_index * 16 + i]);
    } else if (ENC == ENC_Q5_1) {
        const uchar *block = row + (ulong)(k0 / 32) * 24;
        const float d = load_half(block), m = load_half(block + 2);
        const uint high = *(const uint *)(block + 4);
        const uchar *qs = block + 8;
        for (int i = 0; i < 16; ++i) {
            const uint j = i;
            const uint byte = qs[j];
            // elements 0..15 are the low nibbles of the 16 bytes, elements 16..31 the high nibbles
            const uint q = half_index == 0 ? ((byte & 0x0f) | (((high >> j) << 4) & 0x10)) : ((byte >> 4) | ((high >> (j + 12)) & 0x10));
            out[i] = d * float(q) + m;
        }
    } else {
        const uchar *block = row + (ulong)(k0 / 256) * 144;
        const uint sb = (k0 % 256) / 32;  // sub-block of 32 values inside the super-block
        const float d = load_half(block), dmin = load_half(block + 2);
        float scale, minimum;
        scale_min_k4(sb, block + 4, scale, minimum);
        const uchar *qs = block + 16 + 32 * (sb / 2);
        const uint shift = 4 * (sb % 2);
        for (int i = 0; i < 16; ++i) {
            const uint q = (qs[half_index * 16 + i] >> shift) & 0x0f;
            out[i] = d * scale * float(q) - dmin * minimum;
        }
    }
}

// `weights`: the first of 64 rows (row bytes `nb01`), `rows_here` of them real; `x_rows[v]`: the vector of each of 32 slots (null = none);
// `store(row, vector, value)` writes one result.
template <int ENC, typename Store>
__device__ __forceinline__ void mm_tile(const uchar *weights, ulong nb01, int k, int rows_here, const float *const *x_rows, int vectors_here, Store store) {
    __shared__ float sw[64][33];
    __shared__ float sx[32][33];
    const int t = threadIdx.x;
    const int load_row = t / 2, half_index = t % 2;
    const int load_vector = t / 4, part = t % 4;
    const int row_group = t % 16, vector_group = t / 16;
    float acc[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) acc[i][j] = 0;
    const uchar *weight_row = weights + (ulong)(load_row < rows_here ? load_row : 0) * nb01;
    const float *x_row = x_rows[load_vector];
    for (int k0 = 0; k0 < k; k0 += 32) {
        float w16[16];
        load_weights16<ENC>(weight_row, k0, half_index, w16);
        for (int i = 0; i < 16; ++i) sw[load_row][half_index * 16 + i] = load_row < rows_here ? w16[i] : 0.0f;
        for (int i = 0; i < 8; ++i) sx[load_vector][part * 8 + i] = (x_row != nullptr && load_vector < vectors_here) ? x_row[k0 + part * 8 + i] : 0.0f;
        __syncthreads();
        for (int kk = 0; kk < 32; ++kk) {
            float a[4], b[4];
            for (int i = 0; i < 4; ++i) a[i] = sw[row_group * 4 + i][kk];
            for (int j = 0; j < 4; ++j) b[j] = sx[vector_group * 4 + j][kk];
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
    for (int i = 0; i < 4; ++i) {
        const int row = row_group * 4 + i;
        if (row >= rows_here) continue;
        for (int j = 0; j < 4; ++j) {
            const int vector = vector_group * 4 + j;
            if (vector < vectors_here) store(row, vector, acc[i][j]);
        }
    }
}

// Dense: dst[token * ne0 + row]; grid x = tiles of 32 tokens, y = tiles of 64 rows.
template <int ENC>
__device__ __forceinline__ void mm_dense(const uchar *src0, const uchar *src1, float *dst, int64_t ne00, uint64_t nb01, uint64_t nb11, int64_t ne0, int64_t ne1) {
    __shared__ const float *x_rows[32];
    const int r0 = blockIdx.y * 64, r1 = blockIdx.x * 32;
    if (threadIdx.x < 32) {
        const int token = r1 + threadIdx.x;
        x_rows[threadIdx.x] = token < ne1 ? (const float *)(src1 + (ulong)token * nb11) : nullptr;
    }
    __syncthreads();
    const int rows_here = (int)min((int64_t)64, ne0 - r0), vectors_here = (int)min((int64_t)32, ne1 - r1);
    mm_tile<ENC>(src0 + (ulong)r0 * nb01, nb01, (int)ne00, rows_here, x_rows, vectors_here, [&](int row, int vector, float value) {
        dst[(ulong)(r1 + vector) * ne0 + r0 + row] = value;
    });
}

KERNEL void mm_q8_0(const uchar *src0, const uchar *src1, float *dst, int64_t ne00, int64_t ne02, uint64_t nb01, uint64_t nb02, uint64_t nb03,
                    int64_t ne12, uint64_t nb10, uint64_t nb11, uint64_t nb12, uint64_t nb13, int64_t ne0, int64_t ne1, uint r2, uint r3) {
    mm_dense<ENC_Q8_0>(src0, src1, dst, ne00, nb01, nb11, ne0, ne1);
}

// Row lists: one thread per (token, slot) puts itself in the list of the expert it chose.
KERNEL void kernel_moe_route_rows(const int32_t *ids, uint *counts, ushort2_t *rows, int64_t nei0, int64_t nei1, int64_t experts) {
    const int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= nei0 * nei1) return;
    const int32_t id = ids[gid];
    if (id < 0 || id >= experts) return;
    const uint position = atomicAdd(&counts[id], 1u);
    rows[id * nei1 + position] = ushort2_t{(ushort)(gid % nei0), (ushort)(gid / nei0)};
}

// Indexed: expert i02 = blockIdx.z reads its own matrix; the (slot, token) pairs it serves come from the row lists.
template <int ENC>
__device__ __forceinline__ void mm_indexed(const uchar *src0s, const uchar *w_host, uint dev_slots, const uchar *src1, float *dst, const ushort2_t *rows_all,
                                           const uint *counts, int64_t nei1, int64_t ne00, uint64_t nb01, uint64_t nb02, int64_t ne11, uint64_t nb11,
                                           uint64_t nb12, int64_t ne0, int64_t ne1) {
    __shared__ const float *x_rows[32];
    __shared__ ushort2_t ids[32];
    const int i02 = blockIdx.z;
    const uint count = counts[i02];
    const int r0 = blockIdx.y * 64, r1 = blockIdx.x * 32;
    if (r1 >= (int)count) return;
    if (threadIdx.x < 32) {
        const int index = r1 + threadIdx.x;
        if (index < (int)count) {
            const ushort2_t id = rows_all[(int64_t)i02 * nei1 + index];
            ids[threadIdx.x] = id;
            x_rows[threadIdx.x] = (const float *)(src1 + (ulong)id.y * nb12 + (ulong)(id.x % ne11) * nb11);
        } else {
            x_rows[threadIdx.x] = nullptr;
        }
    }
    __syncthreads();
    const int rows_here = (int)min((int64_t)64, ne0 - r0), vectors_here = (int)min((int64_t)32, (int64_t)count - r1);
    mm_tile<ENC>(expert_at(src0s, w_host, dev_slots, i02, nb02) + (ulong)r0 * nb01, nb01, (int)ne00, rows_here, x_rows, vectors_here, [&](int row, int vector, float value) {
        const ushort2_t id = ids[vector];
        dst[(ulong)id.x * ne0 + (ulong)id.y * ne0 * ne1 + r0 + row] = value;
    });
}

#define MM_ID(NAME, ENC)                                                                                                                              \
    KERNEL void NAME(const uchar *src0s, const uchar *src1, float *dst, const ushort2_t *rows_all, const uint *counts, int64_t nei1, int64_t ne00,       \
                     int64_t ne02, uint64_t nb01, uint64_t nb02, int64_t ne11, int64_t ne12, uint64_t nb10, uint64_t nb11, uint64_t nb12, int64_t ne0, \
                     int64_t ne1, const uchar *w_host, uint dev_slots) {                                                                             \
        mm_indexed<ENC>(src0s, w_host, dev_slots, src1, dst, rows_all, counts, nei1, ne00, nb01, nb02, ne11, nb11, nb12, ne0, ne1);                  \
    }
MM_ID(mm_id_q8_0, ENC_Q8_0)
MM_ID(mm_id_q5_1, ENC_Q5_1)
MM_ID(mm_id_q4_k, ENC_Q4_K)
