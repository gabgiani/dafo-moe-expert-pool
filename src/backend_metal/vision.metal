// Kernels of the vision encoder (Gemma 4's patch-based ViT with 2D rotary embeddings). They are compiled after matmul.metal, whose
// matrix-matrix product is instantiated here for half-precision weights (`mm_f16`). Activations are F32, except the queries, keys and
// values of the attention, which are half. The encoder is described in src/vision.rs.

template <typename type4x4>
void dequantize_f16(device const half4x4 *src, short il, thread type4x4 &reg) {
    reg = (type4x4)(*src);
}
template [[host_name("mm_f16")]] kernel mat_mm_t kernel_mul_mm<half, half4x4, simdgroup_half8x8, half4x4, 1, dequantize_f16>;

// out[p, c * ps * ps + dy * ps + dx] = (pixel[c, py * ps + dy, px * ps + dx] - 0.5) * 2   for patch p = py * grid_w + px.
kernel void v_patchify(
    device const float *pixels [[buffer(0)]], device float *out [[buffer(1)]], constant uint &width [[buffer(2)]],
    constant uint &height [[buffer(3)]], constant uint &ps [[buffer(4)]], constant uint &grid_w [[buffer(5)]],
    uint2 position [[thread_position_in_grid]])
{
    const uint column = position.x, patch = position.y;
    const uint per_patch = 3 * ps * ps;
    if (column >= per_patch) return;
    const uint c = column / (ps * ps), rest = column % (ps * ps);
    const uint dy = rest / ps, dx = rest % ps;
    const uint py = patch / grid_w, px = patch % grid_w;
    out[patch * per_patch + column] = (pixels[(c * height + py * ps + dy) * width + px * ps + dx] - 0.5f) * 2.0f;
}

// x[p, :] += table[0][col(p)] + table[1][row(p)]; the table is [2][entries][hidden] F32.
kernel void v_add_position(
    device float *x [[buffer(0)]], device const float *table [[buffer(1)]], constant uint &hidden [[buffer(2)]],
    constant uint &grid_w [[buffer(3)]], constant uint &entries [[buffer(4)]], uint2 position [[thread_position_in_grid]])
{
    if (position.x >= hidden) return;
    const uint patch = position.y;
    const uint column = patch % grid_w, row = patch / grid_w;
    x[patch * hidden + position.x] += table[column * hidden + position.x] + table[(entries + row) * hidden + position.x];
}

// RMS norm of `rows` groups of `width` values (width <= 4096): one SIMD group per group; `has_scale` selects the learned weight.
kernel void v_rms_norm(
    device const float *x [[buffer(0)]], device const float *w [[buffer(1)]], device float *y [[buffer(2)]],
    constant uint &width [[buffer(3)]], constant float &eps [[buffer(4)]], constant uint &has_scale [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]])
{
    x += group * width;
    y += group * width;
    float sum = 0;
    for (uint i = lane; i < width; i += 32) sum += x[i] * x[i];
    sum = simd_sum(sum);
    const float inverse = rsqrt(sum / float(width) + eps);
    for (uint i = lane; i < width; i += 32) y[i] = x[i] * inverse * (has_scale ? w[i] : 1.0f);
}

// x += y
kernel void v_add(device float *x [[buffer(0)]], device const float *y [[buffer(1)]], constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]])
{
    if (i < n) x[i] += y[i];
}

inline float v_gelu_tanh(float x) {
    return 0.5f * x * (1.0f + precise::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

// out[p, i] = gelu(gate[p, i]) * up[p, i]; gate and up have `width` columns per patch, `out` rows are `out_stride` apart (the pad stays zero).
kernel void v_gelu_mul(
    device const float *gate [[buffer(0)]], device const float *up [[buffer(1)]], device float *out [[buffer(2)]],
    constant uint &width [[buffer(3)]], constant uint &in_stride [[buffer(4)]], constant uint &out_stride [[buffer(5)]],
    uint2 position [[thread_position_in_grid]])
{
    if (position.x >= width) return;
    out[position.y * out_stride + position.x] = v_gelu_tanh(gate[position.y * in_stride + position.x]) * up[position.y * in_stride + position.x];
}

// 2D rotary embedding on `heads` heads of `head_dim` values, in place, writing half: the first half of each head is rotated with the
// column of the patch and the second with its row; inside each half, value i is paired with value i + half/2 (rotate_half).
kernel void v_rope(
    device const float *x [[buffer(0)]], device half *y [[buffer(1)]], constant uint &heads [[buffer(2)]],
    constant uint &head_dim [[buffer(3)]], constant uint &grid_w [[buffer(4)]], constant float &theta [[buffer(5)]],
    uint3 position [[thread_position_in_grid]])
{
    const uint pair = position.x, head = position.y, patch = position.z;
    const uint per_axis = head_dim / 2, half_axis = per_axis / 2;
    if (pair >= per_axis) return;                     // pair indexes a (axis, i) with i < half_axis
    const uint axis = pair / half_axis, i = pair % half_axis;
    const float coordinate = axis == 0 ? float(patch % grid_w) : float(patch / grid_w);
    const float angle = coordinate * pow(theta, -2.0f * float(i) / float(per_axis));
    const float c = cos(angle), s = sin(angle);
    const uint base = (patch * heads + head) * head_dim + axis * per_axis;
    const float x1 = x[base + i], x2 = x[base + half_axis + i];
    y[base + i] = half(x1 * c - x2 * s);
    y[base + half_axis + i] = half(x2 * c + x1 * s);
}

// y = half(x)
kernel void v_to_half(device const float *x [[buffer(0)]], device half *y [[buffer(1)]], constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]])
{
    if (i < n) y[i] = half(x[i]);
}

// Bidirectional attention over the `n` real patches (the buffers hold `padded` rows, a multiple of 8): scale 1, no mask. One SIMD group
// handles 8 queries of one head and walks the keys in blocks of 8 with an online softmax; q, k and v are half [padded, heads * D],
// the output is F32 [padded, heads * D]. D = 72 (nine 8x8 tiles).
#define V_D 72
#define V_TILES 9
kernel void v_attention(
    device const half *q [[buffer(0)]], device const half *k [[buffer(1)]], device const half *v [[buffer(2)]],
    device float *out [[buffer(3)]], constant uint &n [[buffer(4)]], constant uint &heads [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float scores[64];
    threadgroup half probabilities[64];
    threadgroup float diagonal[64];
    const uint row_stride = heads * V_D;
    const uint head = group.y, first = group.x * 8;
    device const half *qh = q + first * row_stride + head * V_D;
    device const half *kh = k + head * V_D;
    device const half *vh = v + head * V_D;

    simdgroup_half8x8 qt[V_TILES];
    for (short t = 0; t < V_TILES; ++t) simdgroup_load(qt[t], qh + t * 8, row_stride);
    simdgroup_float8x8 o[V_TILES];
    for (short t = 0; t < V_TILES; ++t) o[t] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    float m = -INFINITY, l = 0.0f;                          // per lane: row (lane / 4) of the 8 queries (4 lanes share a row)
    const uint row = lane / 4, part = lane % 4;
    for (uint key = 0; key < n; key += 8) {
        simdgroup_float8x8 s = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (short t = 0; t < V_TILES; ++t) {
            simdgroup_half8x8 kt;
            simdgroup_load(kt, kh + key * row_stride + t * 8, row_stride, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(s, qt[t], kt, s);
        }
        simdgroup_store(s, scores, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        float values[2];
        float block_max = -INFINITY;
        for (short j = 0; j < 2; ++j) {
            const uint column = part * 2 + j;
            values[j] = key + column < n ? scores[row * 8 + column] : -INFINITY;
            block_max = max(block_max, values[j]);
        }
        block_max = max(block_max, simd_shuffle_xor(block_max, 1));
        block_max = max(block_max, simd_shuffle_xor(block_max, 2));
        const float new_max = max(m, block_max);
        const float alpha = (m == -INFINITY) ? 0.0f : exp(m - new_max);
        float sum = 0.0f;
        for (short j = 0; j < 2; ++j) {
            const float p = values[j] == -INFINITY ? 0.0f : exp(values[j] - new_max);
            probabilities[row * 8 + part * 2 + j] = half(p);
            sum += p;
        }
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 2);
        l = l * alpha + sum;
        m = new_max;
        // diag(alpha) * O + P * V
        for (short i = 0; i < 2; ++i) diagonal[row * 8 + part * 2 + i] = 0.0f;
        if (part == 0) diagonal[row * 8 + row] = alpha;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 d;
        simdgroup_load(d, diagonal, 8);
        simdgroup_half8x8 p_tile;
        simdgroup_load(p_tile, probabilities, 8);
        for (short t = 0; t < V_TILES; ++t) {
            simdgroup_float8x8 scaled = make_filled_simdgroup_matrix<float, 8>(0.0f);
            simdgroup_multiply_accumulate(scaled, d, o[t], scaled);
            simdgroup_half8x8 vt;
            simdgroup_load(vt, vh + key * row_stride + t * 8, row_stride);
            simdgroup_multiply_accumulate(o[t], p_tile, vt, scaled);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    // O / l
    for (short i = 0; i < 2; ++i) diagonal[row * 8 + part * 2 + i] = 0.0f;
    if (part == 0) diagonal[row * 8 + row] = 1.0f / l;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 d;
    simdgroup_load(d, diagonal, 8);
    for (short t = 0; t < V_TILES; ++t) {
        simdgroup_float8x8 result = make_filled_simdgroup_matrix<float, 8>(0.0f);
        simdgroup_multiply_accumulate(result, d, o[t], result);
        simdgroup_store(result, out + first * row_stride + head * V_D + t * 8, row_stride);
    }
}

// Average pooling of the patch grid into 3x3 cells, times sqrt(hidden), then the model's standardization (x - bias) * scale.
// out[cell, :] with cell = (row / k) * (grid_w / k) + (col / k).
kernel void v_pool(
    device const float *x [[buffer(0)]], device const float *bias [[buffer(1)]], device const float *scale [[buffer(2)]],
    device float *out [[buffer(3)]], constant uint &hidden [[buffer(4)]], constant uint &grid_w [[buffer(5)]],
    constant uint &k [[buffer(6)]], constant uint &has_standard [[buffer(7)]], uint2 position [[thread_position_in_grid]])
{
    if (position.x >= hidden) return;
    const uint cells_w = grid_w / k;
    const uint cell = position.y, cx = cell % cells_w, cy = cell / cells_w;
    float sum = 0.0f;
    for (uint dy = 0; dy < k; ++dy)
        for (uint dx = 0; dx < k; ++dx) sum += x[((cy * k + dy) * grid_w + cx * k + dx) * hidden + position.x];
    float value = sum / float(k * k) * sqrt(float(hidden));
    if (has_standard != 0) value = (value - bias[position.x]) * scale[position.x];
    out[cell * hidden + position.x] = value;
}
