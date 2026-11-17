// Kernels of the Metal backend. The compiler writes a header with the constants of the model before this text:
//   NR, NSG      rows per SIMD group and SIMD groups per thread group of the matrix-vector products
// Activations are F32. `state` holds the work in flight: state[0] = position of the first token, state[2] = token chosen by the
// head, state[8 + t] = id of the t-th token of the batch. Every kernel that works per token takes the token index from its grid
// (y or z), so one kernel serves a single token and a batch. state[3] = tokens of the batch that are real (a plan may be larger than
// its batch), state[STATE_VISION + t] = the last position the t-th token may attend to when it belongs to an image (0 otherwise):
// the tokens of one image see each other in both directions in the sliding layers.
// A token id with its top bit set is not a vocabulary row: it names row (id & 0x7fffffff) of the `extra` embeddings (an image's
// soft tokens), which are used as they are.
#include <metal_stdlib>
using namespace metal;

struct half_block_q8_0 {
    half d;
    int8_t qs[32];
};

inline float load_half(device const uchar *p) {
    return float(*(device const half *)p);
}

// ---- embedding ----------------------------------------------------------------------------------------------------

kernel void embed_q8(
    device const uchar *table [[buffer(0)]], device const uint *state [[buffer(1)]], device float *out [[buffer(2)]],
    constant uint &width [[buffer(3)]], constant float &scale [[buffer(4)]], device const float *extra [[buffer(5)]],
    uint2 position [[thread_position_in_grid]])
{
    const uint i = position.x;
    if (i >= width) return;
    const uint token = state[8 + position.y];
    if (token & 0x80000000u) {
        out[position.y * width + i] = extra[(ulong)(token & 0x7fffffffu) * width + i];
        return;
    }
    const ulong row_bytes = (ulong)(width / 32) * 34;
    device const uchar *block = table + (ulong)token * row_bytes + (ulong)(i / 32) * 34;
    out[position.y * width + i] = float(((device const int8_t *)(block + 2))[i % 32]) * load_half(block) * scale;
}

kernel void zero_u32(device uint *buffer [[buffer(0)]], constant uint &n [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    if (i < n) buffer[i] = 0;
}

// ---- normalization ------------------------------------------------------------------------------------------------

// One thread group per group of `width` values; `has_scale` selects the learned scale.
kernel void rms_norm(
    device const float *x [[buffer(0)]], device const float *w [[buffer(1)]], device float *y [[buffer(2)]],
    constant uint &width [[buffer(3)]], constant float &eps [[buffer(4)]], constant uint &has_scale [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    threadgroup float partial[32];
    x += group * width;
    y += group * width;
    float sum = 0;
    for (uint i = t; i < width; i += threads) sum += x[i] * x[i];
    sum = simd_sum(sum);
    if (t % 32 == 0) partial[t / 32] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < 32) {
        float value = t < threads / 32 ? partial[t] : 0.0f;
        value = simd_sum(value);
        if (t == 0) partial[0] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inverse = rsqrt(partial[0] / float(width) + eps);
    for (uint i = t; i < width; i += threads) y[i] = x[i] * inverse * (has_scale ? w[i] : 1.0f);
}

kernel void scale_by(
    device const float *x [[buffer(0)]], device const float *w [[buffer(1)]], device float *y [[buffer(2)]],
    constant uint &n [[buffer(3)]], uint i [[thread_position_in_grid]])
{
    if (i < n) y[i] = x[i] * w[0];
}

kernel void add(
    device const float *a [[buffer(0)]], device const float *b [[buffer(1)]], device float *y [[buffer(2)]],
    constant uint &n [[buffer(3)]], uint i [[thread_position_in_grid]])
{
    if (i < n) y[i] = a[i] + b[i];
}

inline float gelu_tanh(float x) {
    return 0.5f * x * (1.0f + precise::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

// gelu(gate) * up for `slots` vectors; the strides place gate, up and out of each slot.
kernel void gelu_mul(
    device const float *gate [[buffer(0)]], device const float *up [[buffer(1)]], device float *out [[buffer(2)]],
    constant uint &n [[buffer(3)]], constant uint &gate_stride [[buffer(4)]], constant uint &up_stride [[buffer(5)]],
    constant uint &out_stride [[buffer(6)]], uint2 position [[thread_position_in_grid]])
{
    if (position.x >= n) return;
    out[position.y * out_stride + position.x] = gelu_tanh(gate[position.y * gate_stride + position.x]) * up[position.y * up_stride + position.x];
}

// ---- matrix times vector ------------------------------------------------------------------------------------------

// Q8_0 (34-byte blocks: half scale, 32 int8): `rows` rows of `width` values; SIMD group = NR rows.
kernel void mv_q8_0(
    device const half_block_q8_0 *x [[buffer(0)]], device const float *y [[buffer(1)]], device float *dst [[buffer(2)]],
    constant uint &width [[buffer(3)]], constant uint &rows [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]])
{
    const uint nb = width / 32;
    const uint first = (group.x * NSG + simd) * NR;
    if (first >= rows) return;
    device const half_block_q8_0 *base = x + (ulong)first * nb;
    float sums[NR];
    for (int r = 0; r < NR; ++r) sums[r] = 0;
    const uint ix = lane / 4, il = lane % 4;
    device const float *yb = y + ix * 32 + 8 * il;
    float yl[8];
    for (uint ib = ix; ib < nb; ib += 8) {
        for (int i = 0; i < 8; ++i) yl[i] = yb[i];
        for (int r = 0; r < NR; ++r) {
            device const half_block_q8_0 *block = base + (ulong)r * nb + ib;
            device const int8_t *qs = block->qs + 8 * il;
            float sum = 0;
            for (int i = 0; i < 8; ++i) sum += qs[i] * yl[i];
            sums[r] += sum * float(block->d);
        }
        yb += 8 * 32;
    }
    for (int r = 0; r < NR; ++r) {
        const float total = simd_sum(sums[r]);
        if (lane == 0 && first + r < rows) dst[first + r] = total;
    }
}

inline void scale_min_k4(uint j, device const uchar *q, thread float &scale, thread float &minimum) {
    if (j < 4) {
        scale = float(q[j] & 63);
        minimum = float(q[j + 4] & 63);
    } else {
        scale = float((q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4));
        minimum = float((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

// Expert matrices in Q4_K (144-byte blocks of 256 values). Grid: x = row tiles, y = slot. A stacked tensor holds `total`
// matrices of `rows` rows one after the other; slot `s` reads the matrix `ids[s]`. `x_stride` is 0 when every slot reads the
// same vector. Lane layout: 4 blocks in flight, 8 lanes per block; a lane reads 16 bytes of a pair of 32-value groups.
kernel void mv_id_q4_k(
    device const uchar *w [[buffer(0)]], device const uint *ids [[buffer(1)]], device const float *x [[buffer(2)]],
    device float *y [[buffer(3)]], constant uint &width [[buffer(4)]], constant uint &rows [[buffer(5)]],
    constant uint &x_stride [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]])
{
    const uint slot = group.y;
    const uint nb = width / 256;
    const ulong row_bytes = (ulong)nb * 144;
    device const uchar *base = w + (ulong)ids[slot] * rows * row_bytes;
    const uint first = (group.x * NSG + simd) * NR;
    if (first >= rows) return;
    device const float *xv = x + slot * x_stride;
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
            device const uchar *block = base + (ulong)(first + r) * row_bytes + (ulong)ib * 144;
            const float d = load_half(block), dmin = load_half(block + 2);
            float scale_low, min_low, scale_high, min_high;
            scale_min_k4(2 * pair, block + 4, scale_low, min_low);
            scale_min_k4(2 * pair + 1, block + 4, scale_high, min_high);
            const uint4 packed = *(device const uint4 *)(block + 16 + pair * 32 + half_index * 16);
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
// Lane layout: 16 blocks in flight, 2 lanes per block (8 bytes each).
kernel void mv_id_q5_1(
    device const uchar *w [[buffer(0)]], device const uint *ids [[buffer(1)]], device const float *x [[buffer(2)]],
    device float *y [[buffer(3)]], constant uint &width [[buffer(4)]], constant uint &rows [[buffer(5)]],
    constant uint &x_stride [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]])
{
    const uint slot = group.y;
    const uint nb = width / 32;
    const ulong row_bytes = (ulong)nb * 24;
    device const uchar *base = w + (ulong)ids[slot] * rows * row_bytes;
    const uint first = (group.x * NSG + simd) * NR;
    if (first >= rows) return;
    device const float *xv = x + slot * x_stride;
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
            device const uchar *block = base + (ulong)(first + r) * row_bytes + (ulong)ib * 24;
            const float d = load_half(block), m = load_half(block + 2);
            const uint high = *(device const uint *)(block + 4);
            const uint2 packed = *(device const uint2 *)(block + 8 + h * 8);
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

// Expert matrices in Q8_0, indexed like the kernels above.
kernel void mv_id_q8_0(
    device const uchar *w [[buffer(0)]], device const uint *ids [[buffer(1)]], device const float *x [[buffer(2)]],
    device float *y [[buffer(3)]], constant uint &width [[buffer(4)]], constant uint &rows [[buffer(5)]],
    constant uint &x_stride [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]])
{
    const uint slot = group.y;
    const uint nb = width / 32;
    const uint first = (group.x * NSG + simd) * NR;
    if (first >= rows) return;
    device const half_block_q8_0 *base = (device const half_block_q8_0 *)w + (ulong)ids[slot] * rows * nb + (ulong)first * nb;
    device const float *xv = x + slot * x_stride;
    float sums[NR];
    for (int r = 0; r < NR; ++r) sums[r] = 0;
    const uint ix = lane / 4, il = lane % 4;
    float yl[8];
    for (uint ib = ix; ib < nb; ib += 8) {
        for (int i = 0; i < 8; ++i) yl[i] = xv[ib * 32 + 8 * il + i];
        for (int r = 0; r < NR; ++r) {
            device const half_block_q8_0 *block = base + (ulong)r * nb + ib;
            device const int8_t *qs = block->qs + 8 * il;
            float sum = 0;
            for (int i = 0; i < 8; ++i) sum += qs[i] * yl[i];
            sums[r] += sum * float(block->d);
        }
    }
    for (int r = 0; r < NR; ++r) {
        const float total = simd_sum(sums[r]);
        if (lane == 0 && first + r < rows) y[slot * rows + first + r] = total;
    }
}

kernel void experts_sum(
    device const float *in [[buffer(0)]], device const float *weights [[buffer(1)]], device float *out [[buffer(2)]],
    constant uint &width [[buffer(3)]], constant uint &slots [[buffer(4)]], uint2 position [[thread_position_in_grid]])
{
    const uint i = position.x, token = position.y;
    if (i >= width) return;
    float sum = 0;
    for (uint s = 0; s < slots; ++s) sum += weights[token * slots + s] * in[(token * slots + s) * width + i];
    out[token * width + i] = sum;
}

// ---- router -------------------------------------------------------------------------------------------------------

// One thread group of 1024 threads: normalizes the input, scales it, computes the router logits (F32 matrix of `total`
// rows), picks the `used` largest and writes their ids and their softmax weights times the per-expert scale.
kernel void route(
    device const float *x [[buffer(0)]], device const float *router_scale [[buffer(1)]], device const float *matrix [[buffer(2)]],
    device const float *expert_scale [[buffer(3)]], device uint *ids [[buffer(4)]], device float *weights [[buffer(5)]],
    constant uint &width [[buffer(6)]], constant uint &total [[buffer(7)]], constant uint &used [[buffer(8)]],
    constant float &eps [[buffer(9)]], device const uint *state [[buffer(10)]], constant uint &mask_at [[buffer(11)]],
    device atomic_uint *demand [[buffer(12)]], constant uint &demand_at [[buffer(13)]],
    uint token [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]])
{
    x += (ulong)token * width;
    ids += token * used;
    weights += token * used;
    threadgroup float normalized[ROUTE_MAX_WIDTH];
    threadgroup float logits[ROUTE_MAX_EXPERTS];
    threadgroup float raw[ROUTE_MAX_EXPERTS];
    threadgroup float partial[32];
    float sum = 0;
    for (uint i = t; i < width; i += 1024) sum += x[i] * x[i];
    sum = simd_sum(sum);
    if (lane == 0) partial[simd] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < 32) {
        const float value = simd_sum(partial[t]);
        if (t == 0) partial[0] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float factor = rsqrt(partial[0] / float(width) + eps) / sqrt(float(width));
    for (uint i = t; i < width; i += 1024) normalized[i] = x[i] * factor * router_scale[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = simd; row < total; row += 32) {
        float dot = 0;
        for (uint i = lane; i < width; i += 32) dot += normalized[i] * matrix[row * width + i];
        dot = simd_sum(dot);
        // An expert the host has blocked (state[mask_at + row] == 0xffffffff) can never be chosen: the router renormalizes among the others.
        if (lane == 0) {
            raw[row] = dot;
            logits[row] = state[mask_at + row] == 0xffffffffu ? -INFINITY : dot;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Top-`used` selection by the first SIMD group: each lane holds the logits of experts lane, lane + 32, lane + 64, ... (at most 128).
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
            const uint winner = simd_min(mine == best ? index : 0xffffffffu);
            chosen[s] = winner;
            values[s] = best;
            if ((winner & 31u) == lane) {
                for (uint k = 0; k < K; ++k) {
                    if (k == (winner >> 5)) held[k] = -INFINITY;
                }
            }
        }
        // What the router would have chosen with every expert allowed: counted per expert, so a host can see which experts it keeps
        // wanting that are not in the pool.
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
            const uint winner = simd_min(mine == best ? index : 0xffffffffu);
            if (lane == 0) atomic_fetch_add_explicit(&demand[demand_at + winner], 1u, memory_order_relaxed);
            if ((winner & 31u) == lane) {
                for (uint k = 0; k < K; ++k) {
                    if (k == (winner >> 5)) wanted[k] = -INFINITY;
                }
            }
        }
        if (lane == 0) {
            float largest = values[0], denominator = 0;
            for (uint s = 0; s < used; ++s) {
                values[s] = exp(values[s] - largest);
                denominator += values[s];
            }
            for (uint s = 0; s < used; ++s) {
                // The expert products read their weights by slot: 0 = the expert's own index; otherwise the slot of the pool the host
                // loaded it into, plus one.
                const uint word = state[mask_at + chosen[s]];
                ids[s] = word == 0 ? chosen[s] : word - 1;
                weights[s] = values[s] / denominator * expert_scale[chosen[s]];
            }
        }
    }
}

// ---- rotary embedding and cache -----------------------------------------------------------------------------------

// Half-split rotary embedding of `heads` heads; `inverse` holds the head_dim / 2 inverse frequencies.
kernel void rope(
    device const float *in [[buffer(0)]], device float *out [[buffer(1)]], device const float *inverse [[buffer(2)]],
    device const uint *state [[buffer(3)]], constant uint &head_dim [[buffer(4)]], constant uint &heads [[buffer(5)]],
    uint3 position [[thread_position_in_grid]])
{
    const uint half_dim = head_dim / 2;
    if (position.x >= half_dim || position.y >= heads) return;
    const float angle = float(state[0] + position.z) * inverse[position.x];
    const float c = precise::cos(angle), s = precise::sin(angle);
    const uint base = (position.z * heads + position.y) * head_dim;
    const float x1 = in[base + position.x], x2 = in[base + position.x + half_dim];
    out[base + position.x] = x1 * c - x2 * s;
    out[base + position.x + half_dim] = x2 * c + x1 * s;
}

// Writes key and value of the tokens into a cache (half precision) laid out [kv_head][capacity][head_dim], as a ring of `capacity` positions.
// A sliding layer's capacity is its window plus the largest batch, so a batch never overwrites what an earlier token of it still reads.
kernel void cache_write(
    device const float *key [[buffer(0)]], device const float *value [[buffer(1)]], device half *key_cache [[buffer(2)]],
    device half *value_cache [[buffer(3)]], device const uint *state [[buffer(4)]], constant uint &head_dim [[buffer(5)]],
    constant uint &capacity [[buffer(6)]], constant uint &kv_heads [[buffer(7)]], uint3 position [[thread_position_in_grid]])
{
    if (position.x >= head_dim || position.z >= state[3]) return;
    const uint slot = (state[0] + position.z) % capacity;
    const ulong destination = ((ulong)position.y * capacity + slot) * head_dim + position.x;
    const ulong source = ((ulong)position.z * kv_heads + position.y) * head_dim + position.x;
    key_cache[destination] = half(key[source]);
    value_cache[destination] = half(value[source]);
}

// ---- attention ----------------------------------------------------------------------------------------------------

// One thread group (8 SIMD groups) per query head. Each SIMD group walks its share of the positions with an online softmax;
// the groups are merged at the end. HEAD_DIM is the width of a head (defined by the instance below).
#define ATTENTION(NAME, HEAD_DIM)                                                                                       \
kernel void NAME(                                                                                                        \
    device const float *query [[buffer(0)]], device const half *key_cache [[buffer(1)]],                                 \
    device const half *value_cache [[buffer(2)]], device float *out [[buffer(3)]], device const uint *state [[buffer(4)]], \
    constant uint &heads [[buffer(5)]], constant uint &kv_heads [[buffer(6)]], constant uint &capacity [[buffer(7)]],    \
    constant uint &window [[buffer(8)]],                                                                                 \
    uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],                               \
    uint simd [[simdgroup_index_in_threadgroup]])                                                                        \
{                                                                                                                        \
    constexpr uint PER_LANE = HEAD_DIM / 32;                                                                             \
    threadgroup float merged[8 * HEAD_DIM];                                                                              \
    threadgroup float maxima[8], sums[8];                                                                                \
    const uint head = group.x;                                                                                           \
    const uint position = state[0] + group.y;                                                                            \
    uint last = position;                                                                                                \
    if (window > 0) {                                                                                                    \
        const uint image_end = state[STATE_VISION + group.y];                                                            \
        if (image_end > position) last = min(image_end, position + window - 1);                                          \
    }                                                                                                                    \
    const uint begin = window > 0 && position + 1 > window ? position + 1 - window : 0;                                  \
    const ulong base = (ulong)(head / (heads / kv_heads)) * capacity * HEAD_DIM;                                         \
    float q[PER_LANE], accumulator[PER_LANE];                                                                            \
    for (uint j = 0; j < PER_LANE; ++j) {                                                                                \
        q[j] = query[(group.y * heads + head) * HEAD_DIM + lane + 32 * j];                                               \
        accumulator[j] = 0;                                                                                              \
    }                                                                                                                    \
    float maximum = -1e30f, sum = 0;                                                                                     \
    for (uint p = begin + simd; p <= last; p += 8) {                                                                     \
        const ulong offset = base + (ulong)(p % capacity) * HEAD_DIM;                                                    \
        float dot = 0;                                                                                                   \
        for (uint j = 0; j < PER_LANE; ++j) dot += q[j] * float(key_cache[offset + lane + 32 * j]);                      \
        dot = simd_sum(dot);                                                                                             \
        const float next = max(maximum, dot);                                                                            \
        const float shrink = exp(maximum - next), weight = exp(dot - next);                                              \
        sum = sum * shrink + weight;                                                                                     \
        for (uint j = 0; j < PER_LANE; ++j) accumulator[j] = accumulator[j] * shrink + weight * float(value_cache[offset + lane + 32 * j]); \
        maximum = next;                                                                                                  \
    }                                                                                                                    \
    for (uint j = 0; j < PER_LANE; ++j) merged[simd * HEAD_DIM + lane + 32 * j] = accumulator[j];                        \
    if (lane == 0) { maxima[simd] = maximum; sums[simd] = sum; }                                                         \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                                                     \
    if (simd == 0) {                                                                                                     \
        float overall = -1e30f;                                                                                          \
        for (uint s = 0; s < 8; ++s) overall = max(overall, maxima[s]);                                                  \
        float denominator = 0;                                                                                           \
        float factors[8];                                                                                                \
        for (uint s = 0; s < 8; ++s) {                                                                                   \
            factors[s] = exp(maxima[s] - overall);                                                                       \
            denominator += factors[s] * sums[s];                                                                         \
        }                                                                                                                \
        for (uint j = 0; j < PER_LANE; ++j) {                                                                            \
            float value = 0;                                                                                             \
            for (uint s = 0; s < 8; ++s) value += factors[s] * merged[s * HEAD_DIM + lane + 32 * j];                     \
            out[(group.y * heads + head) * HEAD_DIM + lane + 32 * j] = value / denominator;                              \
        }                                                                                                                \
    }                                                                                                                    \
}

ATTENTION(attention_256, 256)
ATTENTION(attention_512, 512)

// ---- head ---------------------------------------------------------------------------------------------------------

// The token with the highest logit (the lowest id on a tie): one thread group of 1024 threads. The softcap is monotonic
// and does not change the choice.
kernel void argmax_token(
    device const float *logits [[buffer(0)]], device uint *state [[buffer(1)]], constant uint &n [[buffer(2)]],
    uint t [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]])
{
    threadgroup float values[32];
    threadgroup uint indices[32];
    float best = -INFINITY;
    uint best_index = 0;
    for (uint i = t; i < n; i += 1024) {
        if (logits[i] > best) {
            best = logits[i];
            best_index = i;
        }
    }
    for (uint offset = 16; offset > 0; offset /= 2) {
        const float other = simd_shuffle_down(best, offset);
        const uint other_index = simd_shuffle_down(best_index, offset);
        if (other > best || (other == best && other_index < best_index)) {
            best = other;
            best_index = other_index;
        }
    }
    if (lane == 0) {
        values[simd] = best;
        indices[simd] = best_index;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
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
