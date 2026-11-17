// Matrix-matrix kernels for prompt processing: simdgroup-matrix products of quantized weights with a batch of activations,
// and the indexed form for mixture-of-experts layers. Ported from llama.cpp (MIT licence, Georgi Gerganov and contributors)
// through the copy in candle-metal-kernels; only the tile code and the encodings Gemma 4 needs are kept.
#define QK_K 256
#define K_SCALE_SIZE 12
#define QK5_1 32
#define QK8_0 32
typedef struct {
    union {
        struct {
            half d; // delta
            half m; // min
        };
        half2 dm;
    };
    uint8_t qh[4];         // 5-th bit of quants
    uint8_t qs[QK5_1 / 2]; // nibbles / quants
} block_q5_1;
typedef struct {
    half d;       // delta
    int8_t  qs[QK8_0]; // quants
} block_q8_0;
// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 4.5 bits per weight
typedef struct {
    union {
        struct {
            half d;    // super-block scale for quantized scales
            half dmin; // super-block scale for quantized mins
        };
        half2 dm;
    };
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];           // 4--bit quants
} block_q4_K;
template <typename type4x4>
void dequantize_q5_1(device const block_q5_1 *xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 4);
    const float d = xb->d;
    const float m = xb->m;
    const ushort mask = il ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)xb->qh);

    const int x_mv = il ? 4 : 0;

    const int gh_mv = il ? 12 : 0;
    const int gh_bk = il ?  0 : 4;

    for (int i = 0; i < 8; i++) {
        // extract the 5-th bits for x0 and x1
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg[i/2][2*(i%2)+0] = d * x0 + m;
        reg[i/2][2*(i%2)+1] = d * x1 + m;
    }
}

template <typename type4x4>
void dequantize_q8_0(device const block_q8_0 *xb, short il, thread type4x4 & reg) {
    device const int8_t * qs = ((device const int8_t *)xb->qs);
    const half d = xb->d;

    for (int i = 0; i < 16; i++) {
        reg[i/4][i%4] = (qs[i + 16*il] * d);
    }
}

static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)), uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

template <typename type4x4>
void dequantize_q4_K(device const block_q4_K *xb, short il, thread type4x4 & reg) {
    device const uchar * q = xb->qs;

    short is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il&1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d   = il < 2 ? xb->d : xb->d / 16.h;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask = il<2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

#define BLOCK_SIZE_M 64 // 8 simdgroup matrices from matrix A
#define BLOCK_SIZE_N 32 // 4 simdgroup matrices from matrix B
#define BLOCK_SIZE_K 32
#define THREAD_MAT_M 4 // each thread take 4 simdgroup matrices from matrix A
#define THREAD_MAT_N 2 // each thread take 2 simdgroup matrices from matrix B
#define THREAD_PER_BLOCK 128
#define THREAD_PER_ROW 2 // 2 thread for each row in matrix A to load numbers
#define THREAD_PER_COL 4 // 4 thread for each row in matrix B to load numbers
#define SG_MAT_SIZE 64 // simdgroup matrix is of shape 8x8
#define SG_MAT_ROW 8

// each block_q contains 16*nl weights
template<typename T, typename T4x4, typename simdgroup_T8x8, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread T4x4 &)>
kernel void kernel_mul_mm(device const  uchar * src0,
                          device const  uchar * src1,
                          device        float * dst,
                          constant    int64_t & ne00,
                          constant    int64_t & ne02,
                          constant   uint64_t & nb01,
                          constant   uint64_t & nb02,
                          constant   uint64_t & nb03,
                          constant    int64_t & ne12,
                          constant   uint64_t & nb10,
                          constant   uint64_t & nb11,
                          constant   uint64_t & nb12,
                          constant   uint64_t & nb13,
                          constant    int64_t & ne0,
                          constant    int64_t & ne1,
                          constant       uint & r2,
                          constant       uint & r3,
                          threadgroup   uchar * shared_memory [[threadgroup(0)]],
                          uint3                 tgpig[[threadgroup_position_in_grid]],
                          uint                  tiitg[[thread_index_in_threadgroup]],
                          uint                  sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup T     * sa = (threadgroup T     *)(shared_memory);
    threadgroup float * sb = (threadgroup float *)(shared_memory + 4096);

    const uint r0 = tgpig.y;
    const uint r1 = tgpig.x;
    const uint im = tgpig.z;

    // if this block is of 64x32 shape or smaller
    short n_rows = (ne0 - r0*BLOCK_SIZE_M < BLOCK_SIZE_M) ? (ne0 - r0*BLOCK_SIZE_M) : BLOCK_SIZE_M;
    short n_cols = (ne1 - r1*BLOCK_SIZE_N < BLOCK_SIZE_N) ? (ne1 - r1*BLOCK_SIZE_N) : BLOCK_SIZE_N;

    // a thread shouldn't load data outside of the matrix
    short thread_row = ((short)tiitg/THREAD_PER_ROW) < n_rows ? ((short)tiitg/THREAD_PER_ROW) : n_rows - 1;
    short thread_col = ((short)tiitg/THREAD_PER_COL) < n_cols ? ((short)tiitg/THREAD_PER_COL) : n_cols - 1;

    simdgroup_T8x8     ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    short il = (tiitg % THREAD_PER_ROW);

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    uint   offset0 = (i12/r2)*nb02 + (i13/r3)*nb03;
    ushort offset1 = il/nl;

    device const block_q * x = (device const block_q *)(src0 + (r0*BLOCK_SIZE_M + thread_row)*nb01 + offset0) + offset1;
    device const float   * y = (device const float   *)(src1
        + nb13 * i13
        + nb12 * i12
        + nb11 * (r1 * BLOCK_SIZE_N + thread_col)
        + nb10 * (BLOCK_SIZE_K / THREAD_PER_COL * (tiitg % THREAD_PER_COL)));

    for (int loop_k = 0; loop_k < ne00; loop_k += BLOCK_SIZE_K) {
        // load data and store to threadgroup memory
        T4x4 temp_a;
        dequantize_func(x, il, temp_a);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll(16)
        for (short i = 0; i < 16; i++) {
            *(sa + SG_MAT_SIZE * ((tiitg/THREAD_PER_ROW/8) \
            +                     (tiitg%THREAD_PER_ROW)*16 + (i/8)*8) \
            +                     (tiitg/THREAD_PER_ROW)%8  + (i&7)*8) = temp_a[i/4][i%4];
        }

        *(threadgroup float2x4 *)(sb + (tiitg % THREAD_PER_COL)*8*32 + 8*(tiitg/THREAD_PER_COL)) = *((device float2x4 *) y);

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2+nl-1)/nl : x;
        y += BLOCK_SIZE_K;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load matrices from threadgroup memory and conduct outer products
        threadgroup T     * lsma = (sa + THREAD_MAT_M*SG_MAT_SIZE*(sgitg%2));
        threadgroup float * lsmb = (sb + THREAD_MAT_N*SG_MAT_SIZE*(sgitg/2));

        #pragma unroll(4)
        for (short ik = 0; ik < BLOCK_SIZE_K / 8; ik++) {
            #pragma unroll(4)
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + SG_MAT_SIZE * i);
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma unroll(2)
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + SG_MAT_SIZE * i);
            }

            lsma += BLOCK_SIZE_M/SG_MAT_ROW * SG_MAT_SIZE;
            lsmb += BLOCK_SIZE_N/SG_MAT_ROW * SG_MAT_SIZE;

            #pragma unroll(8)
            for (short i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }
        }
    }

    if ((r0 + 1) * BLOCK_SIZE_M <= ne0 && (r1 + 1) * BLOCK_SIZE_N <= ne1) {
        device float * C = dst + (BLOCK_SIZE_M * r0 + 32 * (sgitg &  1)) \
                               + (BLOCK_SIZE_N * r1 + 16 * (sgitg >> 1)) * ne0 + im*ne1*ne0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8 * (i%4) + 8 * ne0 * (i/4), ne0);
        }
    } else {
        // block is smaller than 64x32, we should avoid writing data outside of the matrix
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str = ((threadgroup float *) shared_memory) \
                                      + 32 * (sgitg&1) + (16 * (sgitg>>1))*BLOCK_SIZE_M;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*BLOCK_SIZE_M*(i/4), BLOCK_SIZE_M);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < n_cols; j += BLOCK_SIZE_N) {
                device float  * D  = dst + (r0*BLOCK_SIZE_M) + (r1*BLOCK_SIZE_N + j)*ne0 + im*ne1*ne0;
                device float4 * D4 = (device float4 *) D;

                threadgroup float  * C  = temp_str + (j*BLOCK_SIZE_M);
                threadgroup float4 * C4 = (threadgroup float4 *) C;

                int i = 0;
                for (; i < n_rows/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }

                i *= 4;
                for (; i < n_rows; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }
    }
}

// same as kernel_mul_mm_impl, but src1 and dst are accessed via indices stored in rowids
// same as kernel_mul_mm_impl, but src1 and dst are accessed via indices stored in rowids
template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
void kernel_mul_mm_id_impl(
        device const  uchar * src0,
        device const  uchar * src1,
        threadgroup ushort2 * rowids,
        device        float * dst,
        constant    int64_t & ne00,
        constant    int64_t & ne02,
        constant   uint64_t & nb01,
        constant   uint64_t & nb02,
        constant    int64_t & ne11,
        constant    int64_t & ne12,
        constant   uint64_t & nb10,
        constant   uint64_t & nb11,
        constant   uint64_t & nb12,
        constant    int64_t & ne0,
                    int64_t   ne1,
                    int64_t   ne0ne1,
        threadgroup   uchar * shared_memory,
        uint3                 tgpig[[threadgroup_position_in_grid]],
        uint                  tiitg[[thread_index_in_threadgroup]],
        uint                  sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup half  * sa = (threadgroup half  *)(shared_memory);
    threadgroup float * sb = (threadgroup float *)(shared_memory + 4096);

    const uint r0 = tgpig.y;
    const uint r1 = tgpig.x;

    if (r1 * BLOCK_SIZE_N >= ne1) return;

    // if this block is of 64x32 shape or smaller
    short n_rows = (ne0 - r0 * BLOCK_SIZE_M < BLOCK_SIZE_M) ? (ne0 - r0 * BLOCK_SIZE_M) : BLOCK_SIZE_M;
    short n_cols = (ne1 - r1 * BLOCK_SIZE_N < BLOCK_SIZE_N) ? (ne1 - r1 * BLOCK_SIZE_N) : BLOCK_SIZE_N;

    // a thread shouldn't load data outside of the matrix
    short thread_row = ((short)tiitg/THREAD_PER_ROW) < n_rows ? ((short)tiitg/THREAD_PER_ROW) : n_rows - 1;
    short thread_col = ((short)tiitg/THREAD_PER_COL) < n_cols ? ((short)tiitg/THREAD_PER_COL) : n_cols - 1;

    simdgroup_half8x8  ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 c_res[8];
    for (int i = 0; i < 8; i++){
        c_res[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }
    short il = (tiitg % THREAD_PER_ROW);

    ushort offset1 = il/nl;

    threadgroup const auto & id = rowids[r1 * BLOCK_SIZE_N + thread_col];

    device const block_q * x = (device const block_q *)(src0 + (r0 * BLOCK_SIZE_M + thread_row) * nb01) + offset1;
    device const float   * y = (device const float   *)(src1
        + nb12 * id[1]
        + nb11 * (id[0] % ne11)
        + nb10 * (BLOCK_SIZE_K / THREAD_PER_COL * (tiitg % THREAD_PER_COL)));

    for (int loop_k = 0; loop_k < ne00; loop_k += BLOCK_SIZE_K) {
        // load data and store to threadgroup memory
        half4x4 temp_a;
        dequantize_func(x, il, temp_a);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int i = 0; i < 16; i++) {
            *(sa + SG_MAT_SIZE * ((tiitg / THREAD_PER_ROW / 8) \
            +                     (tiitg % THREAD_PER_ROW) * 16 + (i / 8) * 8) \
            +                     (tiitg / THREAD_PER_ROW) % 8  + (i & 7) * 8) = temp_a[i/4][i%4];
        }

        *(threadgroup float2x4 *)(sb + (tiitg % THREAD_PER_COL) * 8 * 32 + 8 * (tiitg / THREAD_PER_COL)) = *((device float2x4 *)y);

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2+nl-1)/nl : x;
        y += BLOCK_SIZE_K;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load matrices from threadgroup memory and conduct outer products
        threadgroup half  * lsma = (sa + THREAD_MAT_M * SG_MAT_SIZE * (sgitg % 2));
        threadgroup float * lsmb = (sb + THREAD_MAT_N * SG_MAT_SIZE * (sgitg / 2));

        for (int ik = 0; ik < BLOCK_SIZE_K / 8; ik++) {
            for (int i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + SG_MAT_SIZE * i);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (int i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + SG_MAT_SIZE * i);
            }

            lsma += BLOCK_SIZE_M / SG_MAT_ROW * SG_MAT_SIZE;
            lsmb += BLOCK_SIZE_N / SG_MAT_ROW * SG_MAT_SIZE;

            for (int i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(c_res[i], mb[i/4], ma[i%4], c_res[i]);
            }
        }
    }

    {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str = ((threadgroup float *)shared_memory) \
                                      + 32 * (sgitg&1) + (16 * (sgitg>>1)) * BLOCK_SIZE_M;
        for (int i = 0; i < 8; i++) {
            simdgroup_store(c_res[i], temp_str + 8 * (i%4) + 8 * BLOCK_SIZE_M * (i/4), BLOCK_SIZE_M);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        device float * C = dst + (BLOCK_SIZE_M * r0);
        if (sgitg == 0) {
            for (int j = tiitg; j < n_cols; j += BLOCK_SIZE_N) {
                threadgroup const auto & jid = rowids[r1 * BLOCK_SIZE_N + j];
                int joff =  jid[0] * ne0 + jid[1] * ne0ne1;
                for (int i = 0; i < n_rows; i++) {
                    *(C + i + joff) = *(temp_str + i + j * BLOCK_SIZE_M);
                }
            }
        }
    }
}


// Row lists for kernel_mul_mm_id_pre: one thread per (token, slot) puts itself in the list of the expert it chose. Replaces the serial
// scan of every id that kernel_mul_mm_id makes in every threadgroup (nei0*nei1 reads each, repeated for every tile of every expert).
// The order inside a list does not matter: each output row is computed on its own and goes to the position its (token, slot) names.
kernel void kernel_moe_route_rows(
        device const int32_t * ids     [[buffer(0)]],
        device atomic_uint   * counts  [[buffer(1)]],
        device ushort2       * rows    [[buffer(2)]],
        constant int64_t     & nei0    [[buffer(3)]],
        constant int64_t     & nei1    [[buffer(4)]],
        constant int64_t     & experts [[buffer(5)]],
        uint gid [[thread_position_in_grid]]) {
    if ((int64_t) gid >= nei0 * nei1) {
        return;
    }
    const int32_t id = ids[gid];
    if (id < 0 || id >= experts) {
        return;
    }
    const uint position = atomic_fetch_add_explicit(&counts[id], 1u, memory_order_relaxed);
    rows[id * nei1 + position] = ushort2((ushort)(gid % nei0), (ushort)(gid / nei0));
}

// kernel_mul_mm_id with the row lists made by kernel_moe_route_rows: a threadgroup past the expert's row count returns at once and the
// others copy only their own 32 row ids to threadgroup memory.
template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
kernel void kernel_mul_mm_id_pre(
        device const   uchar * src0s,
        device const   uchar * src1,
        device         float * dst,
        device const  ushort2 * rows_all,
        device const    uint * counts,
        constant     int64_t & nei1,
        constant     int64_t & ne00,
        constant     int64_t & ne02,
        constant    uint64_t & nb01,
        constant    uint64_t & nb02,
        constant     int64_t & ne11,
        constant     int64_t & ne12,
        constant    uint64_t & nb10,
        constant    uint64_t & nb11,
        constant    uint64_t & nb12,
        constant     int64_t & ne0,
        constant     int64_t & ne1,
        threadgroup    uchar * shared_memory [[threadgroup(0)]],
        uint3                  tgpig[[threadgroup_position_in_grid]],
        uint                   tiitg[[thread_index_in_threadgroup]],
        uint                   sgitg[[simdgroup_index_in_threadgroup]]) {

    const int32_t i02 = tgpig.z;
    tgpig.z = 0;
    const uint count = counts[i02];
    if (tgpig.x * BLOCK_SIZE_N >= count) {
        return;
    }

    device const uchar * src0 = src0s + i02*nb02;
    threadgroup ushort2 * rowids = (threadgroup ushort2 *)(shared_memory + 8192);

    if (tiitg < BLOCK_SIZE_N) {
        const uint index = tgpig.x * BLOCK_SIZE_N + tiitg;
        if (index < count) {
            rowids[index] = rows_all[i02 * nei1 + index];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    kernel_mul_mm_id_impl<block_q, nl, dequantize_func>(
        src0,
        src1,
        rowids,
        dst,
        ne00,
        ne02,
        nb01,
        nb02,
        ne11,
        ne12,
        nb10,
        nb11,
        nb12,
        ne0,
        (int64_t) count,
        ne0*ne1,
        shared_memory,
        tgpig,
        tiitg,
        sgitg);
}

#define QK_NL 16

typedef decltype(kernel_mul_mm<half, half4x4, simdgroup_half8x8, block_q8_0, 2, dequantize_q8_0>) mat_mm_t;
template [[host_name("mm_q8_0")]] kernel mat_mm_t kernel_mul_mm<half, half4x4, simdgroup_half8x8, block_q8_0, 2, dequantize_q8_0>;

typedef decltype(kernel_mul_mm_id_pre<block_q8_0, 2, dequantize_q8_0>) mat_mm_id_pre_t;
template [[host_name("mm_id_q8_0")]] kernel mat_mm_id_pre_t kernel_mul_mm_id_pre<block_q8_0, 2, dequantize_q8_0>;
template [[host_name("mm_id_q5_1")]] kernel mat_mm_id_pre_t kernel_mul_mm_id_pre<block_q5_1, 2, dequantize_q5_1>;
template [[host_name("mm_id_q4_k")]] kernel mat_mm_id_pre_t kernel_mul_mm_id_pre<block_q4_K, QK_NL, dequantize_q4_K>;
