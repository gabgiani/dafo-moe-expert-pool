// Kernels of the vision encoder in CUDA C++ (the same as backend_metal/vision.metal), compiled after kernels.cu: they use its matrix-matrix
// tile (`mm_tile`), which is instantiated here for half-precision weights (`mm_f16`). Activations are F32, except the queries, keys and
// values of the attention, which are half (`ushort`). See src/vision.rs for what the encoder computes.

#define ENC_F16 3

template <>
__device__ __forceinline__ void load_weights16<ENC_F16>(const uchar *row, uint k0, uint half_index, float *out) {
    const ushort *values = (const ushort *)row + k0 + half_index * 16;
    for (int i = 0; i < 16; ++i) out[i] = h2f(values[i]);
}

KERNEL void mm_f16(const uchar *src0, const uchar *src1, float *dst, int64_t ne00, int64_t ne02, uint64_t nb01, uint64_t nb02, uint64_t nb03,
                   int64_t ne12, uint64_t nb10, uint64_t nb11, uint64_t nb12, uint64_t nb13, int64_t ne0, int64_t ne1, uint r2, uint r3) {
    mm_dense<ENC_F16>(src0, src1, dst, ne00, nb01, nb11, ne0, ne1);
}

KERNEL void v_patchify(const float *pixels, float *out, uint width, uint height, uint ps, uint grid_w) {
    const uint column = blockIdx.x * blockDim.x + threadIdx.x, patch = blockIdx.y;
    const uint per_patch = 3 * ps * ps;
    if (column >= per_patch) return;
    const uint c = column / (ps * ps), rest = column % (ps * ps);
    const uint dy = rest / ps, dx = rest % ps;
    const uint py = patch / grid_w, px = patch % grid_w;
    out[(ulong)patch * per_patch + column] = (pixels[((ulong)c * height + py * ps + dy) * width + px * ps + dx] - 0.5f) * 2.0f;
}

KERNEL void v_add_position(float *x, const float *table, uint hidden, uint grid_w, uint entries) {
    const uint column = blockIdx.x * blockDim.x + threadIdx.x, patch = blockIdx.y;
    if (column >= hidden) return;
    const uint c = patch % grid_w, row = patch / grid_w;
    x[(ulong)patch * hidden + column] += table[(ulong)c * hidden + column] + table[((ulong)entries + row) * hidden + column];
}

KERNEL void v_rms_norm(const float *x, const float *w, float *y, uint width, float eps, uint has_scale) {
    const uint lane = threadIdx.x & 31, group = blockIdx.x;
    x += (ulong)group * width;
    y += (ulong)group * width;
    float sum = 0;
    for (uint i = lane; i < width; i += 32) sum += x[i] * x[i];
    sum = simd_sum(sum);
    const float inverse = rsqrtf(sum / float(width) + eps);
    for (uint i = lane; i < width; i += 32) y[i] = x[i] * inverse * (has_scale ? w[i] : 1.0f);
}

KERNEL void v_add(float *x, const float *y, uint n) {
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

KERNEL void v_gelu_mul(const float *gate, const float *up, float *out, uint width, uint in_stride, uint out_stride) {
    const uint column = blockIdx.x * blockDim.x + threadIdx.x, patch = blockIdx.y;
    if (column >= width) return;
    out[(ulong)patch * out_stride + column] = gelu_tanh(gate[(ulong)patch * in_stride + column]) * up[(ulong)patch * in_stride + column];
}

KERNEL void v_rope(const float *x, ushort *y, uint heads, uint head_dim, uint grid_w, float theta) {
    const uint pair = blockIdx.x * blockDim.x + threadIdx.x, head = blockIdx.y, patch = blockIdx.z;
    const uint per_axis = head_dim / 2, half_axis = per_axis / 2;
    if (pair >= per_axis) return;
    const uint axis = pair / half_axis, i = pair % half_axis;
    const float coordinate = axis == 0 ? float(patch % grid_w) : float(patch / grid_w);
    const float angle = coordinate * powf(theta, -2.0f * float(i) / float(per_axis));
    const float c = cosf(angle), s = sinf(angle);
    const ulong base = ((ulong)patch * heads + head) * head_dim + axis * per_axis;
    const float x1 = x[base + i], x2 = x[base + half_axis + i];
    y[base + i] = f2h(x1 * c - x2 * s);
    y[base + half_axis + i] = f2h(x2 * c + x1 * s);
}

KERNEL void v_to_half(const float *x, ushort *y, uint n) {
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = f2h(x[i]);
}

// Bidirectional attention over `n` patches (scale 1, no mask), D = 72 per head. A block takes 32 queries of one head and walks the keys in
// tiles of 32 with an online softmax; 4 threads share a query (18 values each). q, k and v are half [padded, heads * 72]; out is F32.
#define V_D 72
KERNEL void v_attention(const ushort *q, const ushort *k, const ushort *v, float *out, uint n, uint heads) {
    __shared__ float sk[32][V_D + 1];
    __shared__ float sv[32][V_D + 1];
    const uint t = threadIdx.x, query_in_block = t / 4, part = t % 4;
    const uint head = blockIdx.y, query = blockIdx.x * 32 + query_in_block;
    const ulong row_stride = (ulong)heads * V_D;
    float qv[18], acc[18];
    for (int i = 0; i < 18; ++i) {
        qv[i] = query < n ? h2f(q[(ulong)query * row_stride + head * V_D + part * 18 + i]) : 0.0f;
        acc[i] = 0.0f;
    }
    float m = -1e30f, l = 0.0f;
    for (uint key = 0; key < n; key += 32) {
        __syncthreads();
        for (uint index = t; index < 32 * V_D; index += 128) {
            const uint row = index / V_D, column = index % V_D;
            const bool real = key + row < n;
            sk[row][column] = real ? h2f(k[(ulong)(key + row) * row_stride + head * V_D + column]) : 0.0f;
            sv[row][column] = real ? h2f(v[(ulong)(key + row) * row_stride + head * V_D + column]) : 0.0f;
        }
        __syncthreads();
        for (uint j = 0; j < 32; ++j) {
            if (key + j >= n) break;
            float partial = 0.0f;
            for (int i = 0; i < 18; ++i) partial += qv[i] * sk[j][part * 18 + i];
            partial += __shfl_xor_sync(0xffffffffu, partial, 1);
            partial += __shfl_xor_sync(0xffffffffu, partial, 2);
            const float next = fmaxf(m, partial);
            const float shrink = expf(m - next), weight = expf(partial - next);
            l = l * shrink + weight;
            for (int i = 0; i < 18; ++i) acc[i] = acc[i] * shrink + weight * sv[j][part * 18 + i];
            m = next;
        }
    }
    if (query < n) {
        for (int i = 0; i < 18; ++i) out[(ulong)query * row_stride + head * V_D + part * 18 + i] = acc[i] / l;
    }
}

KERNEL void v_pool(const float *x, const float *bias, const float *scale, float *out, uint hidden, uint grid_w, uint k, uint has_standard) {
    const uint column = blockIdx.x * blockDim.x + threadIdx.x, cell = blockIdx.y;
    if (column >= hidden) return;
    const uint cells_w = grid_w / k;
    const uint cx = cell % cells_w, cy = cell / cells_w;
    float sum = 0.0f;
    for (uint dy = 0; dy < k; ++dy)
        for (uint dx = 0; dx < k; ++dx) sum += x[((ulong)(cy * k + dy) * grid_w + cx * k + dx) * hidden + column];
    float value = sum / float(k * k) * sqrtf(float(hidden));
    if (has_standard != 0) value = (value - bias[column]) * scale[column];
    out[(ulong)cell * hidden + column] = value;
}
