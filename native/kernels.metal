#include <metal_stdlib>
using namespace metal;

#if defined(EGEMMA_BF16)
typedef bfloat real;
#elif defined(EGEMMA_HALF)
typedef half real;
#else
typedef float real;
#endif

struct LinearParams { uint M, K, N, bias; };
struct NormParams { uint rows, width, weighted, residual; float eps, scale; };
struct RopeParams { uint count, heads, dim, length, angles; float theta, factor; };
struct AttnParams { uint batch, length, heads, kv, dim; int window; };
struct ElementParams { uint count, width, layers, layer, exact_gelu; float scale; };
struct PoolParams { uint batch, length, width, dim; };

// Reference: model.py L92 and L71-72. The scalar is rounded to the weight dtype.
kernel void embedding(device const int* ids [[buffer(0)]], device const real* w [[buffer(1)]],
                      device real* out [[buffer(2)]], constant ElementParams& p [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    if (i < p.count) out[i] = real(float(w[uint(ids[i / p.width]) * p.width + i % p.width]) * p.scale);
}

// Row-major X[M,K] * W[N,K]^T. Four SIMD groups cover a 32x32 tile.
// Shared-memory staging handles arbitrary edge dimensions; accumulation is fp32.
kernel void linear(device const real* x [[buffer(0)]], device const real* w [[buffer(1)]],
                   device const real* bias [[buffer(2)]], device real* out [[buffer(3)]],
                   constant LinearParams& p [[buffer(4)]],
                   uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup real a[32*32], b[32*32];
    threadgroup float result[32*32];
    simdgroup_float8x8 c[4];
    for (uint i = 0; i < 4; ++i) c[i] = make_filled_simdgroup_matrix<float,8,8>(0);
    uint r = (sg / 2)*16, n = (sg % 2)*16;
    for (uint base = 0; base < p.K; base += 32) {
        for (uint i = tid; i < 32*32; i += 128) {
            uint row = group.y*32 + i/32, k = base + i%32;
            uint col = group.x*32 + i/32;
            a[i] = row < p.M && k < p.K ? x[row*p.K+k] : real(0);
            b[(i%32)*32 + i/32] = col < p.N && k < p.K ? w[col*p.K+k] : real(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 32; k += 8) {
            simdgroup_matrix<real,8,8> ma[2], mb[2];
            simdgroup_load(ma[0], a+r*32+k, 32);
            simdgroup_load(ma[1], a+(r+8)*32+k, 32);
            simdgroup_load(mb[0], b+k*32+n, 32);
            simdgroup_load(mb[1], b+k*32+n+8, 32);
            simdgroup_multiply_accumulate(c[0], ma[0], mb[0], c[0]);
            simdgroup_multiply_accumulate(c[1], ma[0], mb[1], c[1]);
            simdgroup_multiply_accumulate(c[2], ma[1], mb[0], c[2]);
            simdgroup_multiply_accumulate(c[3], ma[1], mb[1], c[3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(c[0], result+r*32+n, 32);
    simdgroup_store(c[1], result+r*32+n+8, 32);
    simdgroup_store(c[2], result+(r+8)*32+n, 32);
    simdgroup_store(c[3], result+(r+8)*32+n+8, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 32*32; i += 128) {
        uint row = group.y*32+i/32, col = group.x*32+i%32;
        if (row < p.M && col < p.N) out[row*p.N+col] = real(result[i]+(p.bias ? float(bias[col]) : 0));
    }
}

// X[M,K] * Wt[K,N] with the weight stored pre-transposed at load time.
// Four SIMD-group positions (2x2) x S K-slices per threadgroup; every SIMD
// group loads its 8x8 operands directly from device memory. Partial tiles are
// summed in a fixed slice order in threadgroup memory (deterministic), then
// bias is added in fp32 and rounded once to the activation dtype.
// Requires N % BN == 0 and K % (8*S) == 0; X/out rows are allocated padded to
// BM, padded rows are row-independent garbage and are never stored.
//
// Epilogues reproduce the unfused kernels' rounding exactly:
//   0 plain   out[M,N]   = round(acc + bias)
//   1 geglu   Wt columns interleave 16-wide gate/up blocks (see engine load);
//             out[M,N/2] = round(round(gelu(round(gate))) * round(up))   (= gelu_mul)
//   2 ple     out[M,N]   = round(round(gelu(round(acc+bias))) * ple[row,layer,col])
struct TileParams { uint M, K, N, bias, epilogue, layers, layer; };

float gelu_tanh(float x) { return 0.5f*x*(1.0f+tanh(0.7978845608028654f*(x+0.044715f*x*x*x))); }

template<uint BM, uint BN, uint S>
void tile_epilogue(threadgroup const float* partial, device const real* bias, device real* out,
                   device const real* aux, constant TileParams& p, uint2 group, uint tid, uint threads);

template<uint BM, uint BN, uint S>
void linear_tiles(device const real* x, device const real* w, device const real* bias, device real* out,
                  device const real* aux, constant TileParams& p, threadgroup float* partial,
                  uint2 group, uint tid, uint sg) {
    constexpr uint SM = BM/2, SN = BN/2, AM = SM/8, AN = SN/8;
    uint pos = sg%4, slice = sg/4;
    uint tm = (pos/2)*SM, tn = (pos%2)*SN, m0 = group.y*BM+tm, n0 = group.x*BN+tn;
    uint span = p.K/S, k_end = (slice+1)*span;
    simdgroup_float8x8 c[AM][AN];
    for (uint i = 0; i < AM; ++i) for (uint j = 0; j < AN; ++j) c[i][j] = make_filled_simdgroup_matrix<float,8,8>(0);
    for (uint k = slice*span; k < k_end; k += 8) {
        simdgroup_matrix<real,8,8> ma[AM], mb[AN];
        for (uint i = 0; i < AM; ++i) simdgroup_load(ma[i], x+ulong(m0+8*i)*p.K+k, p.K);
        for (uint j = 0; j < AN; ++j) simdgroup_load(mb[j], w+ulong(k)*p.N+n0+8*j, p.N);
        for (uint i = 0; i < AM; ++i) for (uint j = 0; j < AN; ++j)
            simdgroup_multiply_accumulate(c[i][j], ma[i], mb[j], c[i][j]);
    }
    for (uint i = 0; i < AM; ++i) for (uint j = 0; j < AN; ++j)
        simdgroup_store(c[i][j], partial+slice*BM*BN+(tm+8*i)*BN+tn+8*j, BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tile_epilogue<BM,BN,S>(partial, bias, out, aux, p, group, tid, 128*S);
}

// Sum S partial BMxBN tiles (fixed order) and apply the epilogue.
template<uint BM, uint BN, uint S>
void tile_epilogue(threadgroup const float* partial, device const real* bias, device real* out,
                   device const real* aux, constant TileParams& p, uint2 group, uint tid, uint threads) {
    if (p.epilogue == 1) {
        uint width = p.N/2;
        for (uint e = tid; e < BM*BN/2; e += threads) {
            uint r = e/(BN/2), j = e%(BN/2), gc = (j/16)*32+j%16, row = group.y*BM+r;
            if (row >= p.M) continue;
            float g = 0, u = 0;
            for (uint s = 0; s < S; ++s) { g += partial[s*BM*BN+r*BN+gc]; u += partial[s*BM*BN+r*BN+gc+16]; }
            real gate = real(gelu_tanh(float(real(g))));
            out[row*width+group.x*(BN/2)+j] = real(float(gate)*float(real(u)));
        }
        return;
    }
    for (uint e = tid; e < BM*BN; e += threads) {
        uint row = group.y*BM+e/BN, col = group.x*BN+e%BN;
        if (row >= p.M) continue;
        float v = 0;
        for (uint s = 0; s < S; ++s) v += partial[s*BM*BN+e];
        real y = real(v+(p.bias ? float(bias[col]) : 0));
        if (p.epilogue == 2)
            y = real(float(real(gelu_tanh(float(y))))*float(aux[(row*p.layers+p.layer)*p.N+col]));
        out[row*p.N+col] = y;
    }
}
#define LINEAR_TILES(NAME, BM, BN, S) \
kernel void NAME(device const real* x [[buffer(0)]], device const real* w [[buffer(1)]], \
                 device const real* bias [[buffer(2)]], device real* out [[buffer(3)]], \
                 device const real* aux [[buffer(4)]], constant TileParams& p [[buffer(5)]], \
                 uint2 group [[threadgroup_position_in_grid]], \
                 uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) { \
    threadgroup float partial[S*BM*BN]; \
    linear_tiles<BM,BN,S>(x, w, bias, out, aux, p, partial, group, tid, sg); \
}
LINEAR_TILES(linear_t32x32s4, 32, 32, 4)
LINEAR_TILES(linear_t32x64s4, 32, 64, 4)


// Per-head q/k/v RMSNorm + NeoX RoPE on the fused QKV projection output.
// One threadgroup per (row, slot); slots are q heads, then k heads, then v
// heads. Rounding matches rms_norm (scale 1) followed by rope. v_norm=0
// copies V unchanged (Gemma 3 has no V norm).
struct QkvParams { uint rows, heads, kv, dim, length, angles, k_transposed; float eps, theta, factor; uint v_norm; };
kernel void qkv_norm_rope(device const real* qkv [[buffer(0)]], device const real* qw [[buffer(1)]],
                          device const real* kw [[buffer(2)]], device real* q [[buffer(3)]],
                          device real* k [[buffer(4)]], device real* v [[buffer(5)]],
                          constant QkvParams& p [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup real normed[1024];
    threadgroup float sums[4];
    uint row = group.y, slot = group.x, width = (p.heads+2*p.kv)*p.dim;
    bool is_q = slot < p.heads, is_k = !is_q && slot < p.heads+p.kv;
    device const real* src = qkv+ulong(row)*width+slot*p.dim;
    device const real* weight = is_q ? qw : kw;
    float sum = 0;
    for (uint j = tid; j < p.dim; j += 128) { float x = float(src[j]); sum += x*x; }
    sum = simd_sum(sum);
    if (lane == 0) sums[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = 1.0f / sqrt((sums[0]+sums[1]+sums[2]+sums[3]) / float(p.dim) + p.eps);
    for (uint j = tid; j < p.dim; j += 128)
        normed[j] = is_q || is_k ? real(float(src[j]) * inv * float(weight[j])) : p.v_norm ? real(float(src[j]) * inv) : src[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint position = row % p.length, half_dim = p.dim/2;
    // K may be written transposed per (batch, kv head): kT[b][h][d][t].
    bool kt = is_k && p.k_transposed;
    ulong dstride = kt ? p.length : 1;
    device real* dst = is_q ? q+(ulong(row)*p.heads+slot)*p.dim
                     : kt ? k+(ulong(row/p.length)*p.kv+slot-p.heads)*p.dim*p.length+position
                     : is_k ? k+(ulong(row)*p.kv+slot-p.heads)*p.dim
                            : v+(ulong(row)*p.kv+slot-p.heads-p.kv)*p.dim;
    for (uint d = tid; d < p.dim; d += 128) {
        if (!is_q && !is_k) { dst[d] = normed[d]; continue; }
        uint f = d % half_dim;
        float invf = f < p.angles ? 1.0f / pow(p.theta, float(2 * f) / float(p.dim)) / p.factor : 0;
        float angle = float(position) * invf;
        real c = real(cos(angle)), s = real(sin(angle));
        real rotated = d < half_dim ? -normed[d + half_dim] : normed[d - half_dim];
        real xc = real(float(normed[d]) * float(c)), rs = real(float(rotated) * float(s));
        dst[d*dstride] = real(float(xc) + float(rs));
    }
}

// Residual RMSNorm chain, one threadgroup (256 threads) per row:
//   r   = round(round(x*inv1*w1) + residual)      (residual optional)
//   r   = round(r * scalar)                        (layer_scalar optional)
//   out = r;  out2 = round(r*inv2*w2)              (second norm optional)
// Matches rms_norm (scale 1) and scale kernels' rounding.
struct ChainParams { uint rows, width, residual, scaled, second; float eps; };
kernel void norm_chain(device const real* x [[buffer(0)]], device const real* w1 [[buffer(1)]],
                       device const real* residual [[buffer(2)]], device const real* scalar [[buffer(3)]],
                       device const real* w2 [[buffer(4)]], device real* out [[buffer(5)]],
                       device real* out2 [[buffer(6)]], constant ChainParams& p [[buffer(7)]],
                       uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sums[8];
    ulong base = ulong(row)*p.width;
    float sum = 0;
    for (uint j = tid; j < p.width; j += 256) { float v = float(x[base+j]); sum += v*v; }
    sum = simd_sum(sum);
    if (lane == 0) sums[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0;
    for (uint i = 0; i < 8; ++i) total += sums[i];
    float inv = 1.0f / sqrt(total / float(p.width) + p.eps);
    float sum2 = 0;
    for (uint j = tid; j < p.width; j += 256) {
        real r = real(float(x[base+j]) * inv * float(w1[j]));
        if (p.residual) r = real(float(r) + float(residual[base+j]));
        if (p.scaled) r = real(float(r) * float(scalar[0]));
        out[base+j] = r;
        sum2 += float(r)*float(r);
    }
    if (!p.second) return;
    sum2 = simd_sum(sum2);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) sums[sg] = sum2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = 0;
    for (uint i = 0; i < 8; ++i) total += sums[i];
    float inv2 = 1.0f / sqrt(total / float(p.width) + p.eps);
    for (uint j = tid; j < p.width; j += 256)
        out2[base+j] = real(float(out[base+j]) * inv2 * float(w2[j]));
}

// Row-major X[M,K] * W[N,K]^T. A deliberately simple 16x16 tiled matmul.
kernel void linear_small(device const real* x [[buffer(0)]], device const real* w [[buffer(1)]],
                   device const real* bias [[buffer(2)]], device real* out [[buffer(3)]],
                   constant LinearParams& p [[buffer(4)]],
                   uint2 group [[threadgroup_position_in_grid]], uint2 t [[thread_position_in_threadgroup]]) {
    threadgroup float a[16][16], b[16][16];
    uint row = group.y * 16 + t.y, col = group.x * 16 + t.x;
    float sum = 0;
    for (uint base = 0; base < p.K; base += 16) {
        a[t.y][t.x] = row < p.M && base + t.x < p.K ? float(x[row * p.K + base + t.x]) : 0;
        uint wr = group.x * 16 + t.y;
        b[t.y][t.x] = wr < p.N && base + t.x < p.K ? float(w[wr * p.K + base + t.x]) : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 16; ++k) sum += a[t.y][k] * b[t.x][k];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < p.M && col < p.N) out[row * p.N + col] = real(sum + (p.bias ? float(bias[col]) : 0));
}

// RMSNorm is fp32 including its weights. Cast before the residual addition,
// matching model.py L20-26, L150, L158, L163 (also for fp16).
kernel void rms_norm(device const real* x [[buffer(0)]], device const real* w [[buffer(1)]],
                     device const real* residual [[buffer(2)]], device real* out [[buffer(3)]],
                     constant NormParams& p [[buffer(4)]],
                     uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sums[256];
    float sum = 0;
    for (uint j = tid; j < p.width; j += 256) {
        float v = float(real(float(x[row * p.width + j]) * p.scale));
        sum += v * v;
    }
    sums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 128; step; step >>= 1) {
        if (tid < step) sums[tid] += sums[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = 1.0f / sqrt(sums[0] / float(p.width) + p.eps);
    for (uint j = tid; j < p.width; j += 256) {
        uint i = row * p.width + j;
        real v = real(float(real(float(x[i]) * p.scale)) * inv * (p.weighted ? float(w[j]) : 1.0f));
        out[i] = p.residual ? real(float(v) + float(residual[i])) : v;
    }
}

// NeoX halves; positions include padding. The proportional suffix has zero
// frequency. Reference: model.py L37-60, L139-140.
kernel void rope(device const real* x [[buffer(0)]], device real* out [[buffer(1)]],
                 constant RopeParams& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count) return;
    uint d = i % p.dim, half_dim = p.dim / 2, f = d % half_dim;
    uint position = (i / (p.heads * p.dim)) % p.length;
    float inv = f < p.angles ? 1.0f / pow(p.theta, float(2 * f) / float(p.dim)) / p.factor : 0;
    float angle = float(position) * inv;
    // HF casts the RoPE table to the activation dtype before multiplying.
    real c = real(cos(angle)), s = real(sin(angle));
    real rotated = d < half_dim ? -x[i + half_dim] : x[i - half_dim];
    real xc = real(float(x[i]) * float(c)), rs = real(float(rotated) * float(s));
    out[i] = real(float(xc) + float(rs));
}

// Online softmax in 8-query x 32-key tiles. No global T x T buffer.
// QK and PV use SIMD-group matrix operations with fp32 accumulators.
kernel void attention(device const real* q [[buffer(0)]], device const real* k [[buffer(1)]],
                      device const real* v [[buffer(2)]], device const int* mask [[buffer(3)]],
                      device real* out [[buffer(4)]], constant AttnParams& p [[buffer(5)]],
                      uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup real aq[8*32], bk[32*32], probability[8*32];
    threadgroup float scores[8*32], diagonal_values[8*8], old[8*32];
    simdgroup_float8x8 accumulator[16];
    threadgroup float maximum[8], denominator[8], alpha[8];
    threadgroup uint begin_key, end_key;
    uint first_row = group.x*8, batch = group.y/p.heads, head = group.y%p.heads;
    uint kv = head/(p.heads/p.kv);
    if (tid == 0) {
        bool all_queries_valid = true;
        for (uint r = 0; r < 8 && first_row+r < p.length; ++r)
            all_queries_valid = all_queries_valid && mask[batch*p.length+first_row+r];
        begin_key = 0; end_key = p.length;
        // Each valid query contains at least its own valid key. Masked queries
        // keep the full range so an all-masked row retains uniform attention.
        if (p.window >= 0 && all_queries_valid) {
            begin_key = uint(max(0, int(first_row)-p.window))/32*32;
            end_key = min(p.length, first_row+7+uint(p.window)+1);
        }
    }
    if (tid < 8) { maximum[tid] = -FLT_MAX; denominator[tid] = 0; }
    for (uint d = 0; d < (p.dim+31)/32; ++d) accumulator[d] = make_filled_simdgroup_matrix<float,8,8>(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint key = begin_key; key < end_key; key += 32) {
        simdgroup_float8x8 product = make_filled_simdgroup_matrix<float,8,8>(0);
        if (first_row+8 <= p.length && key+32 <= p.length && p.dim%8 == 0) {
            for (uint inner = 0; inner < p.dim; inner += 8) {
                simdgroup_matrix<real,8,8> mq, mk;
                simdgroup_load(mq, q+((batch*p.length+first_row)*p.heads+head)*p.dim+inner, p.heads*p.dim);
                simdgroup_load(mk, k+((batch*p.length+key+sg*8)*p.kv+kv)*p.dim+inner, p.kv*p.dim, 0, true);
                simdgroup_multiply_accumulate(product, mq, mk, product);
            }
        } else {
            for (uint d = 0; d < p.dim; d += 32) {
                for (uint i = tid; i < 8*32; i += 128) {
                    uint row = first_row+i/32, column = d+i%32;
                    aq[i] = row < p.length && column < p.dim ? q[((batch*p.length+row)*p.heads+head)*p.dim+column] : real(0);
                }
                for (uint i = tid; i < 32*32; i += 128) {
                    uint row = key+i/32, column = d+i%32;
                    bk[(i%32)*32+i/32] = row < p.length && column < p.dim ? k[((batch*p.length+row)*p.kv+kv)*p.dim+column] : real(0);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint inner = 0; inner < 32; inner += 8) {
                    simdgroup_matrix<real,8,8> mq, mk;
                    simdgroup_load(mq, aq+inner, 32);
                    simdgroup_load(mk, bk+inner*32+sg*8, 32);
                    simdgroup_multiply_accumulate(product, mq, mk, product);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        simdgroup_store(product, scores+sg*8, 32);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = sg; r < 8; r += 4) {
            uint row = first_row+r, column = key+lane;
            bool allowed = row < p.length && column < p.length && mask[batch*p.length+column]
                && (p.window < 0 || abs(int(row)-int(column)) <= p.window);
            float score = allowed ? scores[r*32+lane] : -FLT_MAX;
            float m = max(maximum[r], simd_max(score));
            float rescale = exp(maximum[r]-m);
            float value = column < p.length ? exp(score-m) : 0;
            probability[r*32+lane] = real(value);
            float sum = simd_sum(value);
            if (lane == 0) {
                maximum[r] = m; alpha[r] = rescale;
                denominator[r] = denominator[r]*rescale+sum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 8*8; i += 128) diagonal_values[i] = i/8 == i%8 ? alpha[i/8] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 diagonal;
        simdgroup_load(diagonal, diagonal_values, 8);
        for (uint d = 0; d < p.dim; d += 32) {
            simdgroup_float8x8 rescaled;
            simdgroup_multiply(rescaled, diagonal, accumulator[d/32]);
            accumulator[d/32] = rescaled;
            if (key+32 <= p.length && d+32 <= p.dim) {
                for (uint inner = 0; inner < 32; inner += 8) {
                    simdgroup_matrix<real,8,8> mp, mv;
                    simdgroup_load(mp, probability+inner, 32);
                    simdgroup_load(mv, v+((batch*p.length+key+inner)*p.kv+kv)*p.dim+d+sg*8, p.kv*p.dim);
                    simdgroup_multiply_accumulate(accumulator[d/32], mp, mv, accumulator[d/32]);
                }
            } else {
                for (uint i = tid; i < 32*32; i += 128) {
                    uint row = key+i/32, column = d+i%32;
                    bk[i] = row < p.length && column < p.dim ? v[((batch*p.length+row)*p.kv+kv)*p.dim+column] : real(0);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint inner = 0; inner < 32; inner += 8) {
                    simdgroup_matrix<real,8,8> mp, mv;
                    simdgroup_load(mp, probability+inner, 32);
                    simdgroup_load(mv, bk+inner*32+sg*8, 32);
                    simdgroup_multiply_accumulate(accumulator[d/32], mp, mv, accumulator[d/32]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint d = 0; d < p.dim; d += 32) {
        simdgroup_store(accumulator[d/32], old+sg*8, 32);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 8*32; i += 128) {
            uint row = first_row+i/32, column = d+i%32;
            if (row < p.length && column < p.dim)
                out[((batch*p.length+row)*p.heads+head)*p.dim+column] = real(old[i]/denominator[i/32]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Flash attention v2 for head_dim == 8*ND*C (instantiated for the released
// head_dims 256/512 and the small random-weight test head_dims 16/32) and
// KB-key blocks.
// A threadgroup holds R row groups of 8 queries; each row group uses C SIMD
// groups that split head_dim. Per 32-key block:
//   1. every SIMD group computes partial QK^T over its head_dim slice;
//   2. each SIMD group finishes the fp32 online softmax for 8/C of the rows
//      (sum of the C partials, mask/window, max, exp, denominator) and
//      publishes activation-dtype probabilities and the rescale factors;
//   3. every SIMD group rescales its fp32 output accumulators (only when a
//      row maximum changed) and accumulates P*V for its slice.
// Operands are loaded straight from device memory; tiles that cross the
// sequence end are zero-filled through threadgroup memory. Masking, window,
// all-masked-row behaviour and rounding points (probability -> real,
// out = real(acc/denominator)) match `attention`.
template<typename T>
void load_rows(thread simdgroup_matrix<T,8,8>& m, device const T* base, uint stride, uint rows_valid,
               bool transpose, threadgroup T* stage, uint lane) {
    if (rows_valid >= 8) { simdgroup_load(m, base, stride, ulong2(0,0), transpose); return; }
    for (uint e = lane; e < 64; e += 32) stage[e] = e/8 < rows_valid ? base[(e/8)*stride+e%8] : T(0);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_load(m, stage, 8, ulong2(0,0), transpose);
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

template<typename T>
void load_cols(thread simdgroup_matrix<T,8,8>& m, device const T* base, uint stride, uint cols_valid,
               threadgroup T* stage, uint lane) {
    if (cols_valid >= 8) { simdgroup_load(m, base, stride); return; }
    for (uint e = lane; e < 64; e += 32) stage[e] = e%8 < cols_valid ? base[(e/8)*stride+e%8] : T(0);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_load(m, stage, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

template<uint R, uint C, uint ND, uint KB>
void flash_body(device const real* q, device const real* k, device const real* v, device const int* mask,
                device real* out, constant AttnParams& p, threadgroup float* partial, threadgroup real* probs,
                threadgroup float* diag, threadgroup float* denominators, threadgroup real* stage,
                threadgroup uint* range, uint2 group, uint tid, uint sg, uint lane) {
    constexpr uint RPS = 8/C;  // softmax rows owned by each SIMD group
    constexpr uint KT = KB/8, LC = KB/32;  // key tiles per block, columns per lane
    uint g = sg/C, c = sg%C, batch = group.y/p.heads, head = group.y%p.heads, kvh = head/(p.heads/p.kv);
    uint block_row = group.x*(8*R), first_row = block_row+g*8;
    uint dbase = c*(8*ND);
    uint rows_valid = first_row < p.length ? min(8u, p.length-first_row) : 0;
    threadgroup float* my_partial = partial+sg*8*KB;
    threadgroup float* group_partial = partial+g*C*8*KB;
    threadgroup real* group_probs = probs+g*8*KB;
    threadgroup float* group_diag = diag+g*64;
    threadgroup real* my_stage = stage+sg*64;
    if (tid == 0) {
        bool all_valid = true;
        for (uint r = 0; r < 8*R && block_row+r < p.length; ++r) all_valid = all_valid && mask[batch*p.length+block_row+r];
        range[0] = 0; range[1] = p.length;
        // Each valid query contains its own valid key; masked queries keep the
        // full range so an all-masked row retains uniform attention.
        if (p.window >= 0 && all_valid) {
            range[0] = uint(max(0, int(block_row)-p.window))/KB*KB;
            range[1] = min(p.length, block_row+8*R-1+uint(p.window)+1);
        }
    }
    if (c == 0) for (uint e = lane; e < 64; e += 32) group_diag[e] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint begin_key = range[0], end_key = range[1];
    simdgroup_float8x8 acc[ND];
    for (uint i = 0; i < ND; ++i) acc[i] = make_filled_simdgroup_matrix<float,8,8>(0);
    float maximum[RPS], denominator[RPS];
    for (uint r = 0; r < RPS; ++r) { maximum[r] = -FLT_MAX; denominator[r] = 0; }
    ulong qstride = p.heads*p.dim, kstride = p.kv*p.dim;
    device const real* qbase = q+(ulong(batch*p.length+first_row)*p.heads+head)*p.dim+dbase;
    // K is stored transposed: kT[batch][kv head][dim][length].
    device const real* ktbase = k+(ulong(batch)*p.kv+kvh)*p.dim*p.length+ulong(dbase)*p.length;
    for (uint key = begin_key; key < end_key; key += KB) {
        simdgroup_float8x8 s[KT];
        for (uint j = 0; j < KT; ++j) s[j] = make_filled_simdgroup_matrix<float,8,8>(0);
        if (rows_valid) {
            for (uint i = 0; i < ND; ++i) {
                simdgroup_matrix<real,8,8> mq;
                load_rows(mq, qbase+8*i, qstride, rows_valid, false, my_stage, lane);
                for (uint j = 0; j < KT; ++j) {
                    uint k0 = key+8*j, kvalid = k0 < p.length ? min(8u, p.length-k0) : 0;
                    simdgroup_matrix<real,8,8> mk;
                    load_cols(mk, ktbase+ulong(8*i)*p.length+k0, p.length, kvalid, my_stage, lane);
                    simdgroup_multiply_accumulate(s[j], mq, mk, s[j]);
                }
            }
        }
        for (uint j = 0; j < KT; ++j) simdgroup_store(s[j], my_partial+8*j, KB);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool key_ok[LC];
        for (uint t = 0; t < LC; ++t) {
            uint column = key+32*t+lane;
            key_ok[t] = column < p.length && mask[batch*p.length+min(column, p.length-1)];
        }
        for (uint rr = 0; rr < RPS; ++rr) {
            uint r = c*RPS+rr, row = first_row+r;
            float score[LC], block_max = -FLT_MAX;
            for (uint t = 0; t < LC; ++t) {
                uint column = key+32*t+lane;
                float sc = 0;
                for (uint cc = 0; cc < C; ++cc) sc += group_partial[cc*8*KB+r*KB+32*t+lane];
                bool allowed = row < p.length && key_ok[t] && (p.window < 0 || abs(int(row)-int(column)) <= p.window);
                score[t] = allowed ? sc : -FLT_MAX;
                block_max = max(block_max, score[t]);
            }
            float m = max(maximum[rr], simd_max(block_max));
            float alpha = exp(maximum[rr]-m), sum = 0;
            for (uint t = 0; t < LC; ++t) {
                float value = key+32*t+lane < p.length ? exp(score[t]-m) : 0;
                group_probs[r*KB+32*t+lane] = real(value);
                sum += value;
            }
            denominator[rr] = denominator[rr]*alpha+simd_sum(sum);
            maximum[rr] = m;
            if (lane == 0) group_diag[r*9] = alpha;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool rescale = false;
        for (uint r = 0; r < 8; ++r) rescale = rescale || group_diag[r*9] != 1.0f;
        if (rescale) {
            simdgroup_float8x8 d;
            simdgroup_load(d, group_diag, 8);
            for (uint i = 0; i < ND; ++i) simdgroup_multiply(acc[i], d, acc[i]);
        }
        for (uint kk = 0; kk < KT; ++kk) {
            uint k0 = key+8*kk, kvalid = k0 < p.length ? min(8u, p.length-k0) : 0;
            if (!kvalid) break;
            simdgroup_matrix<real,8,8> mp;
            simdgroup_load(mp, group_probs+8*kk, KB);
            device const real* vbase = v+(ulong(batch*p.length+k0)*p.kv+kvh)*p.dim+dbase;
            for (uint i = 0; i < ND; ++i) {
                simdgroup_matrix<real,8,8> mv;
                load_rows(mv, vbase+8*i, kstride, kvalid, false, my_stage, lane);
                simdgroup_multiply_accumulate(acc[i], mp, mv, acc[i]);
            }
        }
    }
    if (lane == 0) for (uint rr = 0; rr < RPS; ++rr) denominators[g*8+c*RPS+rr] = denominator[rr];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* my_out = my_partial;  // partial scores are dead after the key loop
    for (uint i = 0; i < ND; ++i) {
        simdgroup_store(acc[i], my_out, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = lane; e < 64; e += 32) {
            uint r = e/8, row = first_row+r;
            if (row < p.length)
                out[(ulong(batch*p.length+row)*p.heads+head)*p.dim+dbase+8*i+e%8] = real(my_out[e]/denominators[g*8+r]);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}
#define FLASH(NAME, R, C, ND, KB) \
kernel void NAME(device const real* q [[buffer(0)]], device const real* k [[buffer(1)]], \
                 device const real* v [[buffer(2)]], device const int* mask [[buffer(3)]], \
                 device real* out [[buffer(4)]], constant AttnParams& p [[buffer(5)]], \
                 uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float partial[R*C*8*KB], diag[R*64], denominators[R*8]; \
    threadgroup real probs[R*8*KB], stage[R*C*64]; \
    threadgroup uint range[2]; \
    flash_body<R,C,ND,KB>(q,k,v,mask,out,p,partial,probs,diag,denominators,stage,range,group,tid,sg,lane); \
}
FLASH(flash_r1c4d8k64, 1, 4, 8, 64)
FLASH(flash_r1c8d8k64, 1, 8, 8, 64)
FLASH(flash_r1c2d1k64, 1, 2, 1, 64)
FLASH(flash_r1c2d2k64, 1, 2, 2, 64)

// Scores and their reduction remain fp32 for both activation dtypes.
// Bidirectional inclusive radius |q-k| <= window, key-only padding mask.
kernel void qk(device const real* q [[buffer(0)]], device const real* k [[buffer(1)]],
               device const int* mask [[buffer(2)]], device float* out [[buffer(3)]],
               constant AttnParams& p [[buffer(4)]], uint3 group [[threadgroup_position_in_grid]],
               uint3 t [[thread_position_in_threadgroup]]) {
    threadgroup float a[16][16], b[16][16];
    uint row = group.y * 16 + t.y, col = group.x * 16 + t.x;
    uint batch = group.z / p.heads, head = group.z % p.heads, kv = head / (p.heads / p.kv);
    float sum = 0;
    for (uint base = 0; base < p.dim; base += 16) {
        a[t.y][t.x] = row < p.length && base + t.x < p.dim ? float(q[((batch*p.length+row)*p.heads+head)*p.dim+base+t.x]) : 0;
        uint kr = group.x * 16 + t.y;
        b[t.y][t.x] = kr < p.length && base + t.x < p.dim ? float(k[((batch*p.length+kr)*p.kv+kv)*p.dim+base+t.x]) : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; j < 16; ++j) sum += a[t.y][j] * b[t.x][j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < p.length && col < p.length) {
        bool allowed = mask[batch*p.length+col] && (p.window < 0 || abs(int(row)-int(col)) <= p.window);
        out[(group.z*p.length+row)*p.length+col] = allowed ? sum : -FLT_MAX;
    }
}

kernel void softmax(device const float* scores [[buffer(0)]], device real* out [[buffer(1)]],
                    constant uint& width [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float values[256];
    float m = -FLT_MAX;
    for (uint j = tid; j < width; j += 256) m = max(m, scores[row*width+j]);
    values[tid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step=128; step; step>>=1) {
        if (tid < step) values[tid] = max(values[tid], values[tid+step]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    m = values[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0;
    for (uint j = tid; j < width; j += 256) sum += exp(scores[row*width+j]-m);
    values[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step=128; step; step>>=1) {
        if (tid < step) values[tid] += values[tid+step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint j = tid; j < width; j += 256) out[row*width+j] = real(exp(scores[row*width+j]-m)/values[0]);
}

kernel void pv(device const real* probs [[buffer(0)]], device const real* v [[buffer(1)]],
               device real* out [[buffer(2)]], constant AttnParams& p [[buffer(3)]],
               uint3 group [[threadgroup_position_in_grid]], uint3 t [[thread_position_in_threadgroup]]) {
    threadgroup float a[16][16], b[16][16];
    uint row = group.y*16+t.y, col = group.x*16+t.x;
    uint batch = group.z/p.heads, head = group.z%p.heads, kv = head/(p.heads/p.kv);
    float sum=0;
    for (uint base=0; base<p.length; base+=16) {
        a[t.y][t.x] = row<p.length && base+t.x<p.length ? float(probs[(group.z*p.length+row)*p.length+base+t.x]) : 0;
        b[t.y][t.x] = base+t.y<p.length && col<p.dim ? float(v[((batch*p.length+base+t.y)*p.kv+kv)*p.dim+col]) : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j=0; j<16; ++j) sum += a[t.y][j]*b[j][t.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row<p.length && col<p.dim) out[((batch*p.length+row)*p.heads+head)*p.dim+col] = real(sum);
}

float gelu(float x, uint) {
    return 0.5f*x*(1.0f+tanh(0.7978845608028654f*(x+0.044715f*x*x*x)));
}

kernel void gelu_mul(device const real* gate [[buffer(0)]], device const real* up [[buffer(1)]],
                     device const real* ple [[buffer(2)]], device real* out [[buffer(3)]],
                     constant ElementParams& p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count) return;
    real g = real(gelu(float(gate[i]), p.exact_gelu));
    uint j = p.layers ? ((i/p.width)*p.layers+p.layer)*p.width+i%p.width : i;
    out[i] = real(float(g)*float(p.layers ? ple[j] : up[i]));
}

kernel void scale(device const real* x [[buffer(0)]], device const real* scalar [[buffer(1)]],
                  device real* out [[buffer(2)]], constant uint& count [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i<count) out[i] = real(float(x[i])*float(scalar[0]));
}

kernel void mean_pool(device const real* x [[buffer(0)]], device const int* mask [[buffer(1)]],
                      device float* out [[buffer(2)]], constant PoolParams& p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.batch*p.width) return;
    uint b=i/p.width, d=i%p.width, count=0;
    float sum=0;
    for (uint t=0; t<p.length; ++t) if (mask[b*p.length+t]) { sum+=float(x[(b*p.length+t)*p.width+d]); ++count; }
    // Keep the reduction and division in fp32: a valid long input can have a
    // pooled SUM > 65504 even when its mean and token embeddings are finite.
    out[i] = count ? sum/float(count) : 0;
}

// Normalize the MRL prefix in fp32. Normalize→slice→normalize is equivalent
// up to fp32 rounding; the learned token-wise projection has already run.
kernel void l2_normalize(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
                         constant PoolParams& p [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sums[256];
    float sum=0;
    for (uint j=tid; j<p.dim; j+=256) { float v=float(x[row*p.width+j]); sum+=v*v; }
    sums[tid]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step=128; step; step>>=1) {
        if(tid<step) sums[tid]+=sums[tid+step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float norm=max(sqrt(sums[0]), 1e-12f);
    for (uint j=tid; j<p.dim; j+=256) out[row*p.dim+j]=float(x[row*p.width+j])/norm;
}

// Optional path, compiled only into the Metal 4 library and selected at
// runtime: the multiply uses Apple's Metal Performance Primitives matmul2d,
// which runs on the GPU neural accelerators of M5-class devices. Same
// transposed weights and the same epilogue as linear_tiles; accumulation is
// fp32 (relaxed_precision = false, float destination).
#ifdef EGEMMA_MPP
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
template<int BM, int BN>
void mpp_tiles(device const real* x, device const real* w, device const real* bias, device real* out,
               device const real* aux, constant TileParams& p, threadgroup float* tile_memory,
               uint2 group, uint tid) {
    using namespace mpp::tensor_ops;
    auto X = tensor<device real, dextents<int32_t,2>, tensor_inline>(const_cast<device real*>(x), dextents<int32_t,2>(p.K, p.M));
    auto W = tensor<device real, dextents<int32_t,2>, tensor_inline>(const_cast<device real*>(w), dextents<int32_t,2>(p.N, p.K));
    constexpr auto desc = matmul2d_descriptor(BM, BN, static_cast<int>(dynamic_extent), false, false, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mA = X.slice(0, int(group.y*BM));
    auto mB = W.slice(int(group.x*BN), 0);
    auto acc = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
    op.run(mA, mB, acc);
    auto tile = tensor<threadgroup float, dextents<int32_t,2>, tensor_inline>(tile_memory, dextents<int32_t,2>(BN, BM));
    acc.store(tile);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tile_epilogue<BM,BN,1>(tile_memory, bias, out, aux, p, group, tid, 128);
}
#define MPP_TILES(NAME, BM, BN) \
kernel void NAME(device const real* x [[buffer(0)]], device const real* w [[buffer(1)]], \
                 device const real* bias [[buffer(2)]], device real* out [[buffer(3)]], \
                 device const real* aux [[buffer(4)]], constant TileParams& p [[buffer(5)]], \
                 uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup float tile_memory[BM*BN]; \
    mpp_tiles<BM,BN>(x, w, bias, out, aux, p, tile_memory, group, tid); \
}
MPP_TILES(mpp_t64x64, 64, 64)
MPP_TILES(mpp_t64x32, 64, 32)
#endif
