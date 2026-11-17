// E2: matrix (Q8_0) times vector with two layouts of the same values. NR (rows per SIMD group) and NSG (SIMD groups per
// thread group) are defined by the host before this source.
#include <metal_stdlib>
using namespace metal;

#define QK 32

struct block_q8_0 {
    half d;
    int8_t qs[QK];
};

struct Params {
    uint width;
    uint rows;
};

// Layout of the GGUF (34-byte blocks: scale, then 32 quants). Same scheme as llama.cpp's kernel_mul_mv_q8_0_f32: every
// thread takes 8 quants of a block and reads them one byte at a time.
kernel void mv_q8_gguf(
    device const block_q8_0 *x,
    device const float *y,
    device float *dst,
    constant Params &p,
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]])
{
    const uint nb = p.width / QK;
    const uint first = (group.x * NSG + simd) * NR;
    device const block_q8_0 *rows = x + first * nb;
    float sums[NR];
    for (int r = 0; r < NR; ++r) sums[r] = 0;
    const uint ix = lane / 4, il = lane % 4;
    device const float *yb = y + ix * QK + 8 * il;
    float yl[8];
    for (uint ib = ix; ib < nb; ib += 8) {
        for (int i = 0; i < 8; ++i) yl[i] = yb[i];
        for (int r = 0; r < NR; ++r) {
            device const int8_t *qs = rows[ib + r * nb].qs + 8 * il;
            float sum = 0;
            for (int i = 0; i < 8; ++i) sum += qs[i] * yl[i];
            sums[r] += sum * rows[ib + r * nb].d;
        }
        yb += 8 * 32;
    }
    for (int r = 0; r < NR; ++r) {
        const float total = simd_sum(sums[r]);
        if (lane == 0 && first + r < p.rows) dst[first + r] = total;
    }
}

// Planar layout: the quants of a row are contiguous (`width` bytes, rows one after another) and the scales live in their
// own array. A thread takes QUANTS quants of a block as 4-byte vector loads; a SIMD group covers 32 * QUANTS contiguous bytes
// of each row per step.
#define PLANAR(NAME, QUANTS)                                                                            \
kernel void NAME(                                                                                        \
    device const char *q,                                                                                \
    device const half *d,                                                                                \
    device const float *y,                                                                               \
    device float *dst,                                                                                   \
    constant Params &p,                                                                                  \
    uint3 group [[threadgroup_position_in_grid]],                                                        \
    uint lane [[thread_index_in_simdgroup]],                                                             \
    uint simd [[simdgroup_index_in_threadgroup]])                                                        \
{                                                                                                        \
    const uint per_block = QK / QUANTS;                                                                  \
    const uint nb = p.width / QK;                                                                        \
    const uint first = (group.x * NSG + simd) * NR;                                                      \
    float sums[NR];                                                                                      \
    for (int r = 0; r < NR; ++r) sums[r] = 0;                                                            \
    const uint ix = lane / per_block, il = lane % per_block;                                             \
    for (uint ib = ix; ib < nb; ib += 32 / per_block) {                                                  \
        const uint column = ib * QK + QUANTS * il;                                                       \
        float4 yl[QUANTS / 4];                                                                           \
        for (int i = 0; i < QUANTS / 4; ++i) yl[i] = *(device const packed_float4 *)(y + column + 4 * i); \
        for (int r = 0; r < NR; ++r) {                                                                   \
            const ulong row = first + r;                                                                 \
            device const char *qs = q + row * p.width + column;                                          \
            float sum = 0;                                                                               \
            for (int i = 0; i < QUANTS / 4; ++i) {                                                       \
                sum += dot(float4(char4(*(device const packed_char4 *)(qs + 4 * i))), yl[i]);            \
            }                                                                                            \
            sums[r] += sum * float(d[row * nb + ib]);                                                    \
        }                                                                                                \
    }                                                                                                    \
    for (int r = 0; r < NR; ++r) {                                                                       \
        const float total = simd_sum(sums[r]);                                                           \
        if (lane == 0 && first + r < p.rows) dst[first + r] = total;                                     \
    }                                                                                                    \
}

PLANAR(mv_q8_planar8, 8)
PLANAR(mv_q8_planar16, 16)
PLANAR(mv_q8_planar32, 32)
