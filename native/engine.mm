#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#define ACCELERATE_NEW_LAPACK 1
#include <Accelerate/Accelerate.h>

#include "engine.h"
#include "kernels_source.h"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace egemma {
namespace {
using Shape = std::vector<size_t>;
NSString* ns(const std::string& s) { return [NSString stringWithUTF8String:s.c_str()]; }
std::string str(NSString* s) { return s ? std::string(s.UTF8String) : std::string(); }
void require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
id read_json(const std::filesystem::path& path) {
    NSData* data = [NSData dataWithContentsOfFile:ns(path.string())];
    require(data != nil, "Cannot read " + path.string());
    NSError* error = nil;
    id result = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    require(result != nil, "Invalid JSON in " + path.string() + ": " + str(error.localizedDescription));
    return result;
}
std::string json_string(id value) {
    NSError* error = nil;
    NSData* data = [NSJSONSerialization dataWithJSONObject:value options:NSJSONWritingSortedKeys error:&error];
    require(data != nil, "JSON serialization: " + str(error.localizedDescription));
    return std::string(static_cast<const char*>(data.bytes), data.length);
}
uint32_t integer(NSDictionary* d, NSString* key, uint32_t fallback) {
    id v = d[key];
    if (!v) return fallback;
    require([v isKindOfClass:NSNumber.class], "Expected integer " + str(key));
    double n = [v doubleValue];
    require(n >= 0 && n <= UINT32_MAX && std::floor(n) == n, "Invalid integer " + str(key));
    return static_cast<uint32_t>(n);
}
float number(NSDictionary* d, NSString* key, float fallback) {
    id v = d[key];
    if (!v) return fallback;
    require([v isKindOfClass:NSNumber.class] && std::isfinite([v doubleValue]), "Invalid number " + str(key));
    return [v floatValue];
}
bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size()-suffix.size(), suffix.size(), suffix) == 0;
}
bool tower(const std::string& s) {
    for (auto marker : {"vision_tower.", "embed_vision.", "audio_tower.", "embed_audio."})
        if (s.find(marker) != std::string::npos) return true;
    return false;
}
size_t product(const Shape& shape) {
    size_t n = 1;
    for (size_t d : shape) { require(d > 0 && n <= SIZE_MAX/d, "Invalid tensor shape"); n *= d; }
    return n;
}
struct Mapping {
    int fd = -1;
    size_t size = 0;
    void* data = MAP_FAILED;
    explicit Mapping(const std::filesystem::path& path) {
        fd = open(path.c_str(), O_RDONLY);
        require(fd >= 0, "Cannot open safetensors: " + path.string());
        struct stat st;
        if (fstat(fd, &st) != 0 || st.st_size < 8) { close(fd); throw std::runtime_error("Invalid safetensors size"); }
        size = static_cast<size_t>(st.st_size);
        data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED) { close(fd); throw std::runtime_error("Cannot mmap safetensors"); }
    }
    ~Mapping() { if (data != MAP_FAILED) munmap(data, size); if (fd >= 0) close(fd); }
};
struct Tensor { id<MTLBuffer> buffer = nil; size_t count = 0; };
// bfloat16 storage: round to nearest even, NaN kept quiet.
uint16_t to_bf16(float v) {
    uint32_t b; std::memcpy(&b,&v,4);
    if ((b & 0x7f800000u) == 0x7f800000u) return uint16_t((b >> 16) | ((b & 0xffffu) ? 0x40u : 0u));
    return uint16_t((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);
}
float from_bf16(uint16_t v) { uint32_t b = uint32_t(v) << 16; float f; std::memcpy(&f,&b,4); return f; }
struct Layer { uint32_t dim, kv, ffn, angles; int32_t window; float theta, factor; };
// These structs match kernels.metal; all fields are exactly 32 bits.
struct LinearParams { uint32_t M,K,N,bias; };
struct TileParams { uint32_t M,K,N,bias,epilogue,layers,layer; };
struct QkvParams { uint32_t rows,heads,kv,dim,length,angles,k_transposed; float eps,theta,factor; uint32_t v_norm; };
struct ChainParams { uint32_t rows,width,residual,scaled,second; float eps; };
struct NormParams { uint32_t rows,width,weighted,residual; float eps,scale; };
struct RopeParams { uint32_t count,heads,dim,length,angles; float theta,factor; };
struct AttnParams { uint32_t batch,length,heads,kv,dim; int32_t window; };
struct ElementParams { uint32_t count,width,layers,layer,exact_gelu; float scale; };
struct PoolParams { uint32_t batch,length,width,dim; };
}

std::string device_info() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        require(device != nil, "No Metal device available");
        bool metal4 = false;
        if (@available(macOS 26.0, *)) metal4 = [device supportsFamily:MTLGPUFamilyMetal4];
        return json_string(@{@"device": device.name, @"unified_memory": @(device.hasUnifiedMemory),
            @"max_buffer_bytes": @(device.maxBufferLength),
            @"recommended_working_set_bytes": @(device.recommendedMaxWorkingSetSize),
            @"apple_gpu_family_10": @([device supportsFamily:MTLGPUFamilyApple10]),
            @"metal4_family": @(metal4)});
    }
}

struct Engine::Impl {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    id<MTLLibrary> mpp_library;   // Metal 4 library with the MPP kernels, or nil
    bool use_mpp = false;
    std::string mpp_status = "not built";
    // Projections with at least this many rows use MPP (when enabled). Set in
    // the constructor from measurements (docs/OPTIMIZATION.md, measure 4):
    // fp16 from 64 rows; fp32 never (MPP fp32 was slower at every length).
    // EGEMMA_MPP_MIN_ROWS overrides it for experiments.
    uint32_t mpp_min_rows = UINT32_MAX;
    std::map<std::string, id<MTLComputePipelineState>> pipelines;
    std::map<std::string, Tensor> weights;
    std::vector<Layer> layers;
    uint32_t hidden, heads, ple_width, output, vocab, max_positions;
    float eps;
    bool half, exact_gelu, attention_bias;  // half: 2-byte activations and weights (fp16 or bf16)
    bool bf16 = false;
    void put(void* base, size_t i, float v) const {
        if (!half) static_cast<float*>(base)[i] = v;
        else if (bf16) static_cast<uint16_t*>(base)[i] = to_bf16(v);
        else static_cast<_Float16*>(base)[i] = _Float16(v);
    }
    float rounded(float v) const { return !half ? v : bf16 ? from_bf16(to_bf16(v)) : float(_Float16(v)); }
    // Model family. false: EmbeddingGemma 2 text tower (the original target).
    // true: Gemma 3 bidirectional text model (EmbeddingGemma 300m): RMSNorm
    // scales by (1+w), attention scale query_pre_attn_scalar^-0.5, sliding
    // window radius W//2, no V norm, no PLE / layer_scalar, and the ST Dense
    // modules after pooling (folded into embedding_projection at load).
    bool gemma3 = false, v_norm = true;
    float attention_scale = 1;
    std::vector<std::filesystem::path> dense_modules;
    // Test hook (scripts/validate_v1.py --mutations): disables one Gemma 3
    // rule so the validation can show it detects that error.
    std::string mutation;
    std::string prefix;
    size_t skipped_towers = 0, defaulted_scalars = 0;
    std::map<std::string, size_t> source_dtypes;
    std::mutex mutex;
    Tensor dummy;
    bool profiling = false, dispatch_sampling = false;
    id<MTLCounterSampleBuffer> samples;
    id<MTLCommandBuffer> active_cb;
    std::vector<std::string> dispatch_names;
    std::string label;  // optional per-dispatch role, only used for profiled timings
    std::map<std::string, size_t> dispatch_counts;
    size_t waits = 0, peak_allocated = 0;
    double gpu_seconds = 0, wait_seconds = 0;
    MTLTimestamp cpu_start = 0, gpu_start = 0;
    std::string last_profile = "{}";
    std::vector<id<MTLBuffer>> scratch;
    // Linear weights with N%32==0 and K%32==0 are stored transposed [K,N] for
    // the SIMD-group tile kernels; others keep [N,K] for the bounds-checked kernels.
    std::set<std::string> transposed;
    // During encode, activations are allocated with rows padded to a multiple
    // of 32 so tile kernels can load whole 8-row blocks; counts stay unpadded.
    size_t pad_from = 0, pad_to = 0;
    int carry_slot = -1;  // when >=0, the next allocate() uses this carry buffer
    // Layer outputs carried to the next layer (out, next input norm), double
    // buffered by layer parity so they never alias the per-layer scratch.
    std::vector<id<MTLBuffer>> carry_pool;
    size_t loaded_tensors = 0, fused_weights = 0;
    size_t scratch_cursor = 0;
    bool scratch_enabled = false;

    id<MTLComputeCommandEncoder> start(id<MTLCommandBuffer> cb) {
        active_cb = cb;
        return profiling && !dispatch_sampling ? nil : [cb computeCommandEncoder];
    }

    Tensor allocate(size_t n, size_t element_bytes = 0) {
        size_t bytes = element_bytes ? element_bytes : (half ? 2 : 4);
        require(n > 0 && n <= UINT32_MAX, "Tensor exceeds Metal indexing limit");
        size_t capacity = pad_from ? size_t((unsigned __int128)n*pad_to/pad_from + 1) : n;
        require(capacity <= device.maxBufferLength/bytes, "Tensor exceeds Metal buffer limit");
        id<MTLBuffer> b;
        if (carry_slot >= 0) {
            size_t slot = size_t(carry_slot); carry_slot = -1;
            if (slot >= carry_pool.size()) carry_pool.resize(slot+1);
            if (!carry_pool[slot] || carry_pool[slot].length < capacity*bytes)
                carry_pool[slot] = [device newBufferWithLength:capacity*bytes options:MTLResourceStorageModeShared];
            b = carry_pool[slot];
        } else if (scratch_enabled) {
            size_t slot = scratch_cursor++;
            if (slot >= scratch.size()) scratch.resize(slot+1);
            if (!scratch[slot] || scratch[slot].length < capacity*bytes)
                scratch[slot] = [device newBufferWithLength:capacity*bytes options:MTLResourceStorageModeShared];
            b = scratch[slot];
        } else b = [device newBufferWithLength:capacity*bytes options:MTLResourceStorageModeShared];
        require(b != nil, "Metal buffer allocation failed");
        peak_allocated = std::max(peak_allocated, size_t(device.currentAllocatedSize));
        return {b, n};
    }
    id<MTLComputePipelineState> pipeline(const std::string& name) {
        auto found = pipelines.find(name);
        if (found != pipelines.end()) return found->second;
        id<MTLFunction> fn = [(name.rfind("mpp_",0) == 0 ? mpp_library : library) newFunctionWithName:ns(name)];
        require(fn != nil, "Missing Metal kernel " + name);
        NSError* error = nil;
        id<MTLComputePipelineState> state = [device newComputePipelineStateWithFunction:fn error:&error];
        require(state != nil, "Metal pipeline " + name + ": " + str(error.localizedDescription));
        size_t need = name.rfind("linear_t",0) == 0 ? 512 : 256;
        require(state.maxTotalThreadsPerThreadgroup >= need, "Metal kernel " + name + " needs " + std::to_string(need) + " threads per group");
        pipelines[name] = state;
        return state;
    }
    template<class P>
    void dispatch(id<MTLComputeCommandEncoder> __strong& enc, const std::string& name,
                  std::initializer_list<Tensor> buffers, const P& params, MTLSize groups, MTLSize threads) {
        size_t index = dispatch_names.size();
        if (profiling) {
            require(2*index+1 < samples.sampleCount, "Profiling sample capacity exceeded");
            if (dispatch_sampling) [enc sampleCountersInBuffer:samples atSampleIndex:2*index withBarrier:YES];
            else {
                MTLComputePassDescriptor* pass = [MTLComputePassDescriptor computePassDescriptor];
                pass.sampleBufferAttachments[0].sampleBuffer = samples;
                pass.sampleBufferAttachments[0].startOfEncoderSampleIndex = 2*index;
                pass.sampleBufferAttachments[0].endOfEncoderSampleIndex = 2*index+1;
                enc = [active_cb computeCommandEncoderWithDescriptor:pass];
            }
        }
        dispatch_names.push_back(label.empty() ? name : name+":"+label); ++dispatch_counts[name]; label.clear();
        [enc setComputePipelineState:pipeline(name)];
        NSUInteger slot = 0;
        for (const Tensor& t : buffers) [enc setBuffer:t.buffer offset:0 atIndex:slot++];
        [enc setBytes:&params length:sizeof(P) atIndex:slot];
        [enc dispatchThreadgroups:groups threadsPerThreadgroup:threads];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        if (profiling) {
            if (dispatch_sampling) [enc sampleCountersInBuffer:samples atSampleIndex:2*index+1 withBarrier:YES];
            else { [enc endEncoding]; enc = nil; }
        }
    }
    template<class P>
    void element(id<MTLComputeCommandEncoder> __strong& enc, const std::string& name,
                 std::initializer_list<Tensor> buffers, const P& p, size_t count) {
        dispatch(enc, name, buffers, p, MTLSizeMake((count+255)/256,1,1), MTLSizeMake(256,1,1));
    }
    void finish(id<MTLCommandBuffer> cb, id<MTLComputeCommandEncoder> enc) {
        [enc endEncoding]; [cb commit];
        auto begin = std::chrono::steady_clock::now();
        [cb waitUntilCompleted]; ++waits;
        wait_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        require(cb.status == MTLCommandBufferStatusCompleted, "Metal execution: " + str(cb.error.localizedDescription));
        gpu_seconds += cb.GPUEndTime-cb.GPUStartTime;
        active_cb = nil;
    }
    // epilogue: 0 plain, 1 GELU(gate)*up on interleaved gate/up columns
    // (output N/2), 2 GELU(x)*ple[row,layer,:]. 1 and 2 need transposed weights.
    Tensor linear(id<MTLComputeCommandEncoder> __strong& enc, const Tensor& x, const std::string& name, uint32_t rows,
                  uint32_t epilogue = 0, const Tensor& aux = {}, uint32_t layer = 0) {
        const Tensor& w = weights.at(name+".weight");
        uint32_t k = static_cast<uint32_t>(x.count/rows), n = static_cast<uint32_t>(w.count/k);
        auto found = weights.find(name+".bias");
        Tensor b = found == weights.end() ? dummy : found->second;
        Tensor out = allocate(size_t(rows)*(epilogue == 1 ? n/2 : n));
        LinearParams p{rows,k,n,found != weights.end()};
        if (profiling) label = name.substr(name.find('.', name.find('.')+1)+1);
        if (use_mpp && rows >= mpp_min_rows && transposed.count(name+".weight")) {
            uint32_t bn = n % 64 == 0 ? 64 : 32;
            TileParams tp{rows,k,n,p.bias,epilogue,static_cast<uint32_t>(layers.size()),layer};
            dispatch(enc,bn == 64 ? "mpp_t64x64" : "mpp_t64x32",{x,w,b,out,aux.buffer ? aux : dummy},tp,
                     MTLSizeMake(n/bn,(rows+63)/64,1),MTLSizeMake(128,1,1));
            return out;
        }
        if (transposed.count(name+".weight")) {
            // 32x64 tiles need more rows to fill the GPU; small M prefers 32x32.
            bool wide = rows >= 256 && n % 64 == 0;
            uint32_t bn = wide ? 64 : 32;
            TileParams tp{rows,k,n,p.bias,epilogue,static_cast<uint32_t>(layers.size()),layer};
            dispatch(enc,wide ? "linear_t32x64s4" : "linear_t32x32s4",{x,w,b,out,aux.buffer ? aux : dummy},tp,
                     MTLSizeMake(n/bn,(rows+31)/32,1),MTLSizeMake(512,1,1));
            return out;
        }
        require(epilogue == 0, "Fused linear epilogue requires transposed weights");
        if (rows < 256)
            dispatch(enc,"linear_small",{x,w,b,out},p,MTLSizeMake((n+15)/16,(rows+15)/16,1),MTLSizeMake(16,16,1));
        else
            dispatch(enc,"linear",{x,w,b,out},p,MTLSizeMake((n+31)/32,(rows+31)/32,1),MTLSizeMake(128,1,1));
        return out;
    }
    Tensor norm(id<MTLComputeCommandEncoder> __strong& enc, const Tensor& x, uint32_t width,
                const std::string& name, const Tensor& residual = {}, float scale = 1) {
        Tensor out = allocate(x.count);
        Tensor w = name.empty() ? dummy : weights.at(name);
        NormParams p{static_cast<uint32_t>(x.count/width),width,!name.empty(),residual.buffer!=nil,eps,scale};
        dispatch(enc,"rms_norm",{x,w,residual.buffer ? residual : dummy,out},p,
                 MTLSizeMake(p.rows,1,1),MTLSizeMake(256,1,1));
        return out;
    }
    // Residual RMSNorm chain (see norm_chain in kernels.metal); writes out and,
    // when w2 is given, out2 = RMSNorm(out)*w2.
    void chain(id<MTLComputeCommandEncoder> __strong& enc, const Tensor& x, const std::string& w1, const Tensor& residual,
               const std::string& scalar, const std::string& w2, const Tensor& out, const Tensor& out2) {
        ChainParams p{static_cast<uint32_t>(x.count/hidden),hidden,residual.buffer!=nil,!scalar.empty(),!w2.empty(),eps};
        dispatch(enc,"norm_chain",{x,weights.at(w1),residual.buffer ? residual : dummy,scalar.empty() ? dummy : weights.at(scalar),
                 w2.empty() ? dummy : weights.at(w2),out,out2.buffer ? out2 : dummy},p,MTLSizeMake(p.rows,1,1),MTLSizeMake(256,1,1));
    }
    // Flash attention kernel for a head_dim, or "" for the legacy kernels.
    // Chosen from measurements (docs/OPTIMIZATION.md, measure 3): below 128
    // tokens the legacy qk/softmax/pv path is as fast or faster.
    std::string flash_kernel(uint32_t dim, uint32_t length) const {
        if (length < 128) return "";
        switch (dim) {
            case 256: return "flash_r1c4d8k64";
            case 512: return "flash_r1c8d8k64";
            case 16: return "flash_r1c2d1k64";   // random-weight test shapes
            case 32: return "flash_r1c2d2k64";
            default: return "";
        }
    }
    Tensor carry(size_t n, int slot) { carry_slot = slot; return allocate(n); }
    Tensor rotate(id<MTLComputeCommandEncoder> __strong& enc, const Tensor& x, uint32_t h, uint32_t length, const Layer& l) {
        Tensor out = allocate(x.count);
        RopeParams p{static_cast<uint32_t>(x.count),h,l.dim,length,l.angles,l.theta,l.factor};
        element(enc,"rope",{x,out},p,x.count);
        return out;
    }
    void capture(const std::string& directory, const std::string& name, const Tensor& t, const Shape& shape,
                 NSMutableDictionary* descriptions) {
        if (directory.empty()) return;
        std::filesystem::create_directories(directory);
        std::vector<float> data(t.count);
        if (half && bf16) {
            const uint16_t* src = static_cast<const uint16_t*>(t.buffer.contents);
            std::transform(src,src+t.count,data.begin(),[](uint16_t v){return from_bf16(v);});
        } else if (half) {
            const _Float16* src = static_cast<const _Float16*>(t.buffer.contents);
            std::transform(src,src+t.count,data.begin(),[](auto v){return float(v);});
        } else std::memcpy(data.data(),t.buffer.contents,t.count*sizeof(float));
        std::ofstream f(std::filesystem::path(directory)/(name+".f32"),std::ios::binary);
        f.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(float));
        require(f.good(),"Cannot write capture " + name);
        NSMutableArray* dims=[NSMutableArray array];
        for (size_t n:shape) [dims addObject:@(n)];
        descriptions[ns(name)]=dims;
    }

    // Config derivation mirrors config.py L17-43 and L113-174. Unknown
    // numerical fields are rejected rather than silently approximated.
    void configure(NSDictionary* raw) {
        NSDictionary* cfg=raw[@"text_config"] ?: raw;
        require([cfg isKindOfClass:NSDictionary.class],"Invalid text_config");
        std::set<std::string> known={"vocab_size","hidden_size","intermediate_size","num_hidden_layers","num_attention_heads",
            "num_key_value_heads","head_dim","hidden_activation","max_position_embeddings","rms_norm_eps",
            "pad_token_id","eos_token_id","bos_token_id","rope_parameters","attention_bias","sliding_window",
            "layer_types","hidden_size_per_layer_input","embedding_dim","sliding_window_pattern","global_head_dim",
            "num_global_key_value_heads","per_layer_config","model_type","architectures","transformers_version",
            "dtype","torch_dtype","initializer_range","attention_dropout","use_cache","tie_word_embeddings",
            "_name_or_path","id2label","label2id","return_dict","output_hidden_states","output_attentions",
            "chunk_size_feed_forward","is_encoder_decoder","problem_type","tokenizer_class","prefix",
            "task_specific_params","_attn_implementation_autoset"};
        gemma3=[cfg[@"model_type"] isEqual:@"gemma3_text"];
        if(gemma3) {
            // Gemma3TextConfig fields (transformers models/gemma3/configuration_gemma3.py).
            known.insert({"_sliding_window_pattern","attn_logit_softcapping","final_logit_softcapping","query_pre_attn_scalar",
                          "rope_local_base_freq","rope_scaling","rope_theta","use_bidirectional_attention"});
            require([cfg[@"use_bidirectional_attention"] boolValue],"Only bidirectional Gemma 3 (EmbeddingGemma) is supported");
            for(NSString* key in @[@"attn_logit_softcapping",@"final_logit_softcapping",@"rope_scaling"])
                require(!cfg[key] || cfg[key]==id(NSNull.null),"Unsupported Gemma 3 option " + str(key));
            if(const char* m=getenv("EGEMMA_TEST_MUTATION")) mutation=m;
        }
        for(NSString* key in cfg) require(known.count(str(key)),"Unrecognized text config key " + str(key));
        hidden=integer(cfg,@"hidden_size",512); heads=integer(cfg,@"num_attention_heads",4);
        ple_width=gemma3 ? 0 : integer(cfg,@"hidden_size_per_layer_input",512);
        output=gemma3 ? hidden : integer(cfg,@"embedding_dim",768);
        if(gemma3) {
            // modeling_gemma3.py L307: scaling = query_pre_attn_scalar**-0.5 (1/16 for head_dim 256).
            attention_scale=std::pow(number(cfg,@"query_pre_attn_scalar",256),-0.5f);
            v_norm=mutation=="v_norm";
            if(mutation=="attn_scale") attention_scale=1;
        }
        vocab=integer(cfg,@"vocab_size",262144); eps=number(cfg,@"rms_norm_eps",1e-6);
        max_positions=integer(cfg,@"max_position_embeddings",262144);
        uint32_t count=integer(cfg,@"num_hidden_layers",24);
        uint32_t pattern=integer(cfg,gemma3 ? @"_sliding_window_pattern" : @"sliding_window_pattern",6);
        require(hidden && heads && (ple_width || gemma3) && output && vocab && count && pattern && eps>0,"Invalid model dimensions");
        require(size_t(count)*ple_width<=UINT32_MAX,"PLE dimension overflow");
        NSString* act=cfg[@"hidden_activation"] ?: @"gelu_pytorch_tanh";
        require([act isEqual:@"gelu_pytorch_tanh"] || [act isEqual:@"gelu_tanh"],"Only the released model's tanh GELU is supported");
        exact_gelu=false; attention_bias=[cfg[@"attention_bias"] boolValue];
        NSArray* types=cfg[@"layer_types"];
        if(types==id(NSNull.null)) types=nil;
        if(types) require([types isKindOfClass:NSArray.class] && types.count==count,"Invalid layer_types");
        NSDictionary* overrides=cfg[@"per_layer_config"];
        if(overrides==id(NSNull.null)) overrides=nil;
        if(overrides) {
            require([overrides isKindOfClass:NSDictionary.class],"Invalid per_layer_config");
            for(NSString* key in overrides) {
                NSScanner* scanner=[NSScanner scannerWithString:key]; NSInteger idx;
                require([scanner scanInteger:&idx] && scanner.isAtEnd && idx>=0 && idx<count,"Invalid layer override index");
                NSDictionary* ov=overrides[key];
                require([ov isKindOfClass:NSDictionary.class],"Invalid layer override");
                for(NSString* name in ov) require([@[@"head_dim",@"num_key_value_heads",@"intermediate_size"] containsObject:name],"Unsupported layer override " + str(name));
            }
        }
        NSDictionary* rope=cfg[@"rope_parameters"];
        if(rope==id(NSNull.null)) rope=nil;
        if(rope) require([rope isKindOfClass:NSDictionary.class],"Invalid rope_parameters");
        for(uint32_t i=0;i<count;++i) {
            NSString* type=types ? types[i] : ((i+1)%pattern ? @"sliding_attention" : @"full_attention");
            if(i+1==count) type=@"full_attention";
            bool full=[type isEqual:@"full_attention"];
            require(full || [type isEqual:@"sliding_attention"],"Unsupported attention type");
            NSDictionary* ov=nil;
            for(NSString* key in overrides) if([key integerValue]==i) ov=overrides[key];
            uint32_t d=integer(cfg,@"head_dim",256), kv=integer(cfg,@"num_key_value_heads",2);
            if(!overrides && full && !gemma3) { d=integer(cfg,@"global_head_dim",512); kv=integer(cfg,@"num_global_key_value_heads",1); }
            d=integer(ov,@"head_dim",d); kv=integer(ov,@"num_key_value_heads",kv);
            uint32_t ffn=integer(ov,@"intermediate_size",integer(cfg,@"intermediate_size",2048));
            require(d && !(d%2) && kv && heads%kv==0 && ffn,"Invalid attention/MLP dimensions");
            NSDictionary* rp=rope ? rope[type] : nil;
            require(!rope || [rp isKindOfClass:NSDictionary.class],"Missing RoPE configuration");
            NSString* rt=rp[@"rope_type"] ?: @"default";
            require([rt isEqual:@"default"] || [rt isEqual:@"proportional"],"Unsupported RoPE type");
            float theta=number(rp,@"rope_theta",full ? 1e6 : 1e4), factor=1;
            // Gemma 3: rope_theta for full layers, rope_local_base_freq for sliding layers.
            if(gemma3 && !rope) theta=full ? number(cfg,@"rope_theta",1e6) : number(cfg,@"rope_local_base_freq",1e4);
            uint32_t angles=d/2;
            if([rt isEqual:@"proportional"]) {
                float partial=number(rp,@"partial_rotary_factor",1);
                require(partial>=0 && partial<=1,"Invalid partial_rotary_factor");
                angles=static_cast<uint32_t>(partial*d/2); factor=number(rp,@"factor",1);
            }
            require(theta>0 && factor>0,"Invalid RoPE theta/factor");
            uint32_t window=integer(cfg,@"sliding_window",512);
            require(window<=INT32_MAX,"Sliding window overflow");
            // Kernels allow |i-j| <= window. EmbeddingGemma 2 masks |i-j| <= W (ARCHITECTURE.md
            // section 4). Gemma 3 bidirectional replaces W by W//2+1 (configuration_gemma3.py
            // L105-106) and masks |i-j| < W//2+1 (modeling_gemma3.py L460-468): radius W//2.
            // The test mutation shifts the radius by one.
            if(gemma3) window=window/2+(mutation=="window" ? 1 : 0);
            layers.push_back({d,kv,ffn,angles,full ? -1 : static_cast<int32_t>(window),theta,factor});
        }
    }
    void load_weights(const std::filesystem::path& directory) {
        std::set<std::filesystem::path> shards;
        auto index=directory/"model.safetensors.index.json";
        if(std::filesystem::exists(index)) {
            NSDictionary* map=read_json(index)[@"weight_map"];
            require([map isKindOfClass:NSDictionary.class],"Invalid safetensors index");
            for(NSString* key in map) {
                std::filesystem::path p=str(map[key]);
                require(!p.is_absolute() && p.lexically_normal().string().find("..") == std::string::npos,"Invalid shard path");
                shards.insert(directory/p);
            }
        } else shards.insert(directory/"model.safetensors");
        struct Descriptor { std::shared_ptr<Mapping> file; NSDictionary* desc; size_t start; };
        std::map<std::string,Descriptor> tensors;
        std::set<std::string> prefixes;
        for(const auto& path:shards) {
            auto f=std::make_shared<Mapping>(path);
            uint64_t header_bytes; std::memcpy(&header_bytes,f->data,8);
            require(header_bytes<=f->size-8 && header_bytes<=100000000,"Invalid safetensors header length");
            NSData* data=[NSData dataWithBytes:static_cast<const char*>(f->data)+8 length:header_bytes];
            NSError* error=nil;
            NSDictionary* header=[NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
            require([header isKindOfClass:NSDictionary.class],"Invalid safetensors header JSON");
            for(NSString* key in header) {
                std::string name=str(key); if(name=="__metadata__") continue;
                require(tensors.emplace(name,Descriptor{f,header[key],size_t(header_bytes)+8}).second,"Duplicate tensor " + name);
                if(ends_with(name,"embed_tokens.weight") && !tower(name)) prefixes.insert(name.substr(0,name.size()-19));
            }
        }
        require(prefixes.size()==1,"Cannot identify unique text weight prefix"); prefix=*prefixes.begin();
        std::map<std::string,Shape> expected;
        expected["embed_tokens.weight"]={vocab,hidden}; expected["norm.weight"]={hidden};
        if(!gemma3) {
            expected["embedding_projection.weight"]={output,hidden};
            expected["ple.per_layer_model_projection.weight"]={layers.size()*ple_width,hidden};
            expected["ple.per_layer_projection_norm.weight"]={ple_width};
        }
        for(size_t i=0;i<layers.size();++i) {
            const auto& l=layers[i]; std::string p="layers."+std::to_string(i)+".";
            expected[p+"self_attn.q_proj.weight"]={heads*l.dim,hidden};
            expected[p+"self_attn.k_proj.weight"]={l.kv*l.dim,hidden};
            expected[p+"self_attn.v_proj.weight"]={l.kv*l.dim,hidden};
            expected[p+"self_attn.o_proj.weight"]={hidden,heads*l.dim};
            if(attention_bias) for(auto proj:{"q_proj","k_proj","v_proj","o_proj"})
                expected[p+"self_attn."+proj+".bias"]={expected.at(p+"self_attn."+proj+".weight")[0]};
            for(auto n:{"q_norm","k_norm"}) expected[p+"self_attn."+n+".weight"]={l.dim};
            for(auto n:{"gate_proj","up_proj"}) expected[p+"mlp."+n+".weight"]={l.ffn,hidden};
            expected[p+"mlp.down_proj.weight"]={hidden,l.ffn};
            for(auto n:{"input_layernorm","post_attention_layernorm","pre_feedforward_layernorm","post_feedforward_layernorm"}) expected[p+n+".weight"]={hidden};
            if(gemma3) continue;
            expected[p+"layer_scalar"]={1};
            expected[p+"ple_block.per_layer_input_gate.weight"]={ple_width,hidden};
            expected[p+"ple_block.per_layer_projection.weight"]={hidden,ple_width};
            expected[p+"ple_block.post_per_layer_input_norm.weight"]={hidden};
        }
        for(const auto& item:tensors) {
            const std::string& name=item.first;
            if(name.compare(0,prefix.size(),prefix)!=0) {
                require(tower(name),"Unexpected non-text tensor " + name); ++skipped_towers; continue;
            }
            std::string rel=name.substr(prefix.size()); auto want=expected.find(rel);
            require(want!=expected.end(),"Unexpected text tensor " + rel);
            const auto& d=item.second; NSArray* dims=d.desc[@"shape"]; Shape shape;
            require([dims isKindOfClass:NSArray.class],"Missing tensor shape");
            for(NSNumber* v in dims) shape.push_back(v.unsignedLongLongValue);
            require(shape==want->second,"Shape mismatch for " + rel);
            size_t n=product(shape); NSString* dtype=d.desc[@"dtype"];
            size_t bytes=[dtype isEqual:@"F32"] ? 4 : 2;
            require([dtype isEqual:@"F32"] || [dtype isEqual:@"F16"] || [dtype isEqual:@"BF16"],"Unsupported weight dtype");
            NSArray* offsets=d.desc[@"data_offsets"];
            require([offsets isKindOfClass:NSArray.class] && offsets.count==2,"Invalid data_offsets");
            uint64_t begin=[offsets[0] unsignedLongLongValue],end=[offsets[1] unsignedLongLongValue];
            require(begin<=end && end<=d.file->size-d.start && n<=SIZE_MAX/bytes && end-begin==n*bytes,"Invalid tensor data length");
            const char* src=static_cast<const char*>(d.file->data)+d.start+begin;
            Tensor t=allocate(n);
            bool linear_weight=false;
            for(auto suffix:{"_proj.weight","per_layer_input_gate.weight","per_layer_projection.weight","per_layer_model_projection.weight","embedding_projection.weight"})
                linear_weight=linear_weight || ends_with(rel,suffix);
            size_t rows_n=shape[0], cols_k=shape.size()==2 ? shape[1] : 1;
            bool transpose=linear_weight && shape.size()==2 && rows_n%32==0 && cols_k%32==0;
            if(transpose) transposed.insert(rel);
            // Gemma 3 RMSNorm computes x*(1+w) (modeling_gemma3.py L148): stored as 1+w.
            // The attention scale is folded into q_norm (exact for the power of two 1/16).
            float add=0, mul=1;
            if(gemma3 && (ends_with(rel,"norm.weight") || ends_with(rel,"layernorm.weight"))) {
                if(mutation!="norm_plus_one") add=1;
                if(ends_with(rel,"q_norm.weight")) mul=attention_scale;
            }
            for(size_t j=0;j<n;++j) {
                float value;
                if(bytes==4) std::memcpy(&value,src+4*j,4);
                else if([dtype isEqual:@"F16"]) { _Float16 v; std::memcpy(&v,src+2*j,2); value=float(v); }
                else { uint16_t v; std::memcpy(&v,src+2*j,2); uint32_t bits=uint32_t(v)<<16; std::memcpy(&value,&bits,4); }
                value=(value+add)*mul;
                size_t to=transpose ? (j%cols_k)*rows_n+j/cols_k : j;
                put(t.buffer.contents,to,value);
            }
            weights.emplace(rel,t); ++source_dtypes[str(dtype)]; ++loaded_tensors;
        }
        for(const auto& item:expected) if(!weights.count(item.first)) {
            require(ends_with(item.first,"layer_scalar"),"Missing text tensor " + item.first);
            Tensor t=allocate(1);
            put(t.buffer.contents,0,1);
            weights.emplace(item.first,t); ++defaulted_scalars;
        }
        if(defaulted_scalars) std::cerr << "[weights] " << defaulted_scalars << " missing layer_scalar buffers defaulted to 1\n";
        if(gemma3) fold_dense();
        fuse_weights();
    }
    // The ST Dense modules after mean pooling (bias-free, identity activation)
    // compose to one linear map; applied per token before pooling it is the same
    // map, so it becomes embedding_projection (the EmbeddingGemma 2 path).
    void fold_dense() {
        std::vector<double> fold(size_t(hidden)*hidden,0.0);  // [out, in] in fp64, starts as identity
        for(uint32_t i=0;i<hidden;++i) fold[size_t(i)*hidden+i]=1;
        uint32_t width=hidden;
        if(mutation=="dense") dense_modules.clear();
        for(const auto& dir:dense_modules) {
            Mapping f(dir/"model.safetensors");
            uint64_t header_bytes; std::memcpy(&header_bytes,f.data,8);
            require(header_bytes<=f.size-8,"Invalid Dense safetensors header");
            NSDictionary* header=[NSJSONSerialization JSONObjectWithData:[NSData dataWithBytes:static_cast<const char*>(f.data)+8 length:header_bytes] options:0 error:nil];
            NSDictionary* d=header[@"linear.weight"];
            require([d isKindOfClass:NSDictionary.class] && header.count<=2,"Dense module must hold exactly linear.weight (no bias)");
            NSArray* dims=d[@"shape"]; NSArray* offsets=d[@"data_offsets"];
            require([d[@"dtype"] isEqual:@"F32"] && dims.count==2 && [dims[1] unsignedIntValue]==width,"Dense weight must be F32 [out, previous width]");
            uint32_t out=[dims[0] unsignedIntValue];
            uint64_t begin=[offsets[0] unsignedLongLongValue], end=[offsets[1] unsignedLongLongValue];
            require(end-begin==uint64_t(out)*width*4 && end<=f.size-8-header_bytes,"Invalid Dense data length");
            const float* w=reinterpret_cast<const float*>(static_cast<const char*>(f.data)+8+header_bytes+begin);
            std::vector<double> wd(w,w+size_t(out)*width), next(size_t(out)*hidden);
            cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,int(out),int(hidden),int(width),1.0,wd.data(),int(width),
                        fold.data(),int(hidden),0.0,next.data(),int(hidden));  // next = W * fold
            fold.swap(next); width=out;
        }
        output=width;
        Tensor t=allocate(size_t(output)*hidden);
        bool transpose=output%32==0 && hidden%32==0;
        for(size_t j=0;j<fold.size();++j) {
            size_t to=transpose ? (j%hidden)*output+j/hidden : j;
            put(t.buffer.contents,to,float(fold[j]));
        }
        weights.emplace("embedding_projection.weight",t);
        if(transpose) transposed.insert("embedding_projection.weight");
    }
    // Load-time layout: concatenate transposed q|k|v columns (and biases) into
    // one projection, and interleave gate/up in 16-column blocks so a tile's
    // epilogue sees matching gate and up columns. Rows stay K-major.
    void fuse_weights() {
        size_t es = half ? 2 : 4;
        auto take = [&](const std::string& name) { Tensor t = weights.at(name); weights.erase(name); transposed.erase(name); return t; };
        for(size_t i=0;i<layers.size();++i) {
            const auto& l=layers[i]; std::string p="layers."+std::to_string(i)+".";
            std::string q=p+"self_attn.q_proj.weight", k=p+"self_attn.k_proj.weight", v=p+"self_attn.v_proj.weight";
            if(transposed.count(q) && transposed.count(k) && transposed.count(v) && l.dim<=1024) {
                size_t K=hidden, widths[3]={size_t(heads)*l.dim,size_t(l.kv)*l.dim,size_t(l.kv)*l.dim}, total=widths[0]+2*widths[1];
                Tensor parts[3]={take(q),take(k),take(v)}, fused=allocate(K*total);
                char* dst=static_cast<char*>(fused.buffer.contents);
                for(size_t kk=0;kk<K;++kk) for(size_t j=0,off=0;j<3;off+=widths[j++])
                    std::memcpy(dst+(kk*total+off)*es,static_cast<char*>(parts[j].buffer.contents)+kk*widths[j]*es,widths[j]*es);
                std::string name=p+"self_attn.qkv_proj";
                weights.emplace(name+".weight",fused); transposed.insert(name+".weight"); ++fused_weights;
                if(attention_bias) {
                    Tensor bias=allocate(total); char* bd=static_cast<char*>(bias.buffer.contents);
                    size_t off=0; for(auto proj:{"q_proj","k_proj","v_proj"}) {
                        Tensor b=take(p+"self_attn."+proj+".bias"); std::memcpy(bd+off*es,b.buffer.contents,b.count*es); off+=b.count; }
                    weights.emplace(name+".bias",bias);
                }
            }
            std::string g=p+"mlp.gate_proj.weight", u=p+"mlp.up_proj.weight";
            if(transposed.count(g) && transposed.count(u) && l.ffn%16==0) {
                size_t K=hidden, F=l.ffn;
                Tensor gate=take(g), up=take(u), fused=allocate(K*2*F);
                char* dst=static_cast<char*>(fused.buffer.contents);
                for(size_t kk=0;kk<K;++kk) for(size_t blk=0;blk<F/16;++blk) {
                    std::memcpy(dst+(kk*2*F+32*blk)*es,static_cast<char*>(gate.buffer.contents)+(kk*F+16*blk)*es,16*es);
                    std::memcpy(dst+(kk*2*F+32*blk+16)*es,static_cast<char*>(up.buffer.contents)+(kk*F+16*blk)*es,16*es);
                }
                weights.emplace(p+"mlp.gateup_proj.weight",fused); transposed.insert(p+"mlp.gateup_proj.weight"); ++fused_weights;
            }
        }
    }
    void setup_matmul(std::string mode, const std::string& source) {
        if (mode == "auto" && getenv("EGEMMA_MATMUL")) mode = getenv("EGEMMA_MATMUL");
        require(mode == "auto" || mode == "native" || mode == "mpp", "matmul must be auto, native or mpp");
        if (mode == "native") { mpp_status = "disabled"; return; }
        mpp_min_rows = half ? 64 : UINT32_MAX;
        if (getenv("EGEMMA_MPP_MIN_ROWS")) mpp_min_rows = uint32_t(std::stoul(getenv("EGEMMA_MPP_MIN_ROWS")));
#ifdef EGEMMA_ENABLE_MPP
        bool supported = false;
        if (@available(macOS 26.0, *)) supported = [device supportsFamily:MTLGPUFamilyMetal4];
        bool accelerated = [device supportsFamily:MTLGPUFamilyApple10];
        if (!supported) mpp_status = "Metal 4 unsupported";
        else if (mode == "auto" && !accelerated) mpp_status = "not an Apple10-family GPU";
        else if (@available(macOS 26.0, *)) {
            MTLCompileOptions* opts=[MTLCompileOptions new]; opts.mathMode=MTLMathModeSafe;
            opts.mathFloatingPointFunctions=MTLMathFloatingPointFunctionsPrecise;
            opts.languageVersion=MTLLanguageVersion4_0;
            NSError* error=nil;
            mpp_library=[device newLibraryWithSource:ns("#define EGEMMA_MPP\n"+source) options:opts error:&error];
            if (mpp_library) { use_mpp = true; mpp_status = "enabled"; }
            else mpp_status = "compile failed: " + str(error.localizedDescription);
        }
#endif
        require(use_mpp || mode != "mpp", "MPP matmul unavailable: " + mpp_status);
    }
    explicit Impl(const std::string& directory,const std::string& dtype,bool profile,const std::string& matmul) : profiling(profile) {
        require(dtype=="fp32" || dtype=="fp16" || dtype=="bf16","dtype must be fp32, fp16 or bf16");
        half=dtype!="fp32"; bf16=dtype=="bf16";
        device=MTLCreateSystemDefaultDevice(); require(device!=nil,"No Metal device");
        queue=[device newCommandQueue]; require(queue!=nil,"Cannot create Metal queue");
        MTLCompileOptions* opts=[MTLCompileOptions new]; opts.mathMode=MTLMathModeSafe;
        opts.mathFloatingPointFunctions=MTLMathFloatingPointFunctionsPrecise;
        // bfloat needs Metal Shading Language 3.1 or later.
        if (bf16) opts.languageVersion=MTLLanguageVersion3_1;
        std::string source=(bf16 ? "#define EGEMMA_BF16\n" : half ? "#define EGEMMA_HALF\n" : "")+std::string(kMetalSource);
        NSError* error=nil;
        library=[device newLibraryWithSource:ns(source) options:opts error:&error];
        require(library!=nil,"Metal source compilation: " + str(error.localizedDescription));
        setup_matmul(matmul, source);
        configure(read_json(std::filesystem::path(directory)/"config.json"));
        // results/eval/v1_fp16_check.json: the residual stream reaches 1.2e5 (> fp16 max).
        require(!gemma3 || !half || bf16,"EmbeddingGemma 300m (Gemma 3) activations overflow fp16; use dtype bf16 or fp32");
        // Transformer, mean Pooling, [Dense...] (Gemma 3 only), Normalize.
        NSArray* modules=read_json(std::filesystem::path(directory)/"modules.json");
        require([modules isKindOfClass:NSArray.class] && modules.count>=3,"Expected Transformer, mean Pooling, Normalize");
        NSUInteger last=modules.count-1;
        require(ends_with(str(modules[0][@"type"]),"Transformer") && ends_with(str(modules[1][@"type"]),"Pooling")
                && ends_with(str(modules[last][@"type"]),"Normalize"),"Unsupported ST module order/type");
        for(NSUInteger i=2;i<last;++i) {
            require(gemma3 && ends_with(str(modules[i][@"type"]),"Dense"),"Unsupported ST module " + str(modules[i][@"type"]));
            auto dir=std::filesystem::path(directory)/str(modules[i][@"path"]);
            NSDictionary* dense=read_json(dir/"config.json");
            require(![dense[@"bias"] boolValue] && [dense[@"activation_function"] hasSuffix:@"Identity"],
                    "Only bias-free Dense modules with identity activation are supported");
            dense_modules.push_back(dir);
        }
        NSDictionary* pool=read_json(std::filesystem::path(directory)/str(modules[1][@"path"])/"config.json");
        // ST >= 5 writes pooling_mode; older configs (EmbeddingGemma 300m) set pooling_mode_mean_tokens.
        bool mean=[pool[@"pooling_mode"] isEqual:@"mean"] || (!pool[@"pooling_mode"] && [pool[@"pooling_mode_mean_tokens"] boolValue]);
        for(NSString* other in @[@"pooling_mode_cls_token",@"pooling_mode_max_tokens",@"pooling_mode_mean_sqrt_len_tokens",
                                 @"pooling_mode_weightedmean_tokens",@"pooling_mode_lasttoken"]) mean=mean && ![pool[other] boolValue];
        require(mean && [pool[@"include_prompt"] boolValue],"Only mean pooling with include_prompt=true is supported");
        load_weights(directory); dummy=allocate(1); std::memset(dummy.buffer.contents,0,half ? 2 : 4);
        if (use_mpp) for (auto name:{"mpp_t64x64","mpp_t64x32"}) pipeline(name);
        for(auto name:{"flash_r1c4d8k64","flash_r1c8d8k64","flash_r1c2d1k64","flash_r1c2d2k64","embedding","linear","linear_small","linear_t32x32s4","linear_t32x64s4","qkv_norm_rope","norm_chain","attention","rms_norm","rope","qk","softmax","pv","gelu_mul","scale","mean_pool","l2_normalize"}) pipeline(name);
        if (profiling) {
            dispatch_sampling = [device supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary];
            require(dispatch_sampling || [device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary], "GPU timestamp sampling unsupported");
            MTLCounterSampleBufferDescriptor* descriptor = [MTLCounterSampleBufferDescriptor new];
            for (id<MTLCounterSet> set in device.counterSets)
                if ([set.name isEqual:MTLCommonCounterSetTimestamp]) descriptor.counterSet = set;
            require(descriptor.counterSet != nil, "GPU timestamp counter set missing");
            descriptor.storageMode = MTLStorageModeShared; descriptor.sampleCount = 4096;
            samples = [device newCounterSampleBufferWithDescriptor:descriptor error:&error];
            require(samples != nil, "GPU timestamp buffer: " + str(error.localizedDescription));
        }
    }
    std::string info() const {
        std::string hardware=device_info();
        NSMutableDictionary* d=[NSJSONSerialization JSONObjectWithData:[NSData dataWithBytes:hardware.data() length:hardware.size()] options:NSJSONReadingMutableContainers error:nil];
        d[@"dtype"]=bf16 ? @"bf16" : half ? @"fp16" : @"fp32"; d[@"matmul_backend"]=use_mpp ? @"mpp" : @"native"; d[@"mpp_status"]=ns(mpp_status);
        d[@"mpp_min_rows"]=@(mpp_min_rows); d[@"text_tensors"]=@(loaded_tensors); d[@"fused_weights"]=@(fused_weights);
        d[@"weight_prefix"]=ns(prefix); d[@"skipped_tower_tensors"]=@(skipped_towers);
        d[@"missing_layer_scalars_defaulted"]=@(defaulted_scalars); d[@"attention_accumulation"]=@"fp32";
        d[@"model_family"]=gemma3 ? @"gemma3" : @"embeddinggemma2"; d[@"folded_dense_modules"]=@(dense_modules.size());
        if(!mutation.empty()) d[@"test_mutation"]=ns(mutation);
        d[@"metal_math"]=@"safe/precise"; d[@"runtime_compilation"]=@YES;
        NSMutableDictionary* types=[NSMutableDictionary dictionary];
        for(const auto& item:source_dtypes) types[ns(item.first)]=@(item.second);
        d[@"source_dtypes"]=types;
        return json_string(d);
    }

    std::vector<float> encode(const int32_t* ids,const int32_t* mask,size_t batch,size_t length,size_t dim,const std::string& directory) {
        std::lock_guard<std::mutex> lock(mutex);
        scratch_enabled = false;
        dispatch_names.clear(); dispatch_counts.clear(); waits = 0; gpu_seconds = 0; wait_seconds = 0;
        peak_allocated = device.currentAllocatedSize;
        if (profiling) [device sampleTimestamps:&cpu_start gpuTimestamp:&gpu_start];
        require(ids && mask && batch>0 && length>0,"Nonempty ids and attention_mask are required");
        require(length<=max_positions && batch<=UINT32_MAX/length,"Input exceeds model/indexing limit");
        require(dim>0 && dim<=output,"Invalid output dimension");
        size_t rows=batch*length;
        require(rows<=UINT32_MAX/(layers.size()*std::max(ple_width,1u)) && rows<=UINT32_MAX/heads/layers[0].dim,"Input tensor indexing overflow");
        require(batch<=UINT32_MAX/heads/length/length,"Attention scores exceed indexing limit");
        for(size_t i=0;i<rows;++i) {
            require(ids[i]>=0 && uint32_t(ids[i])<vocab,"Token ID out of vocabulary");
            require(mask[i]==0 || mask[i]==1,"Attention mask must contain only 0 or 1");
        }
        pad_from=rows; pad_to=(rows+31)/32*32;
        Tensor idbuf=allocate(rows,4),maskbuf=allocate(rows,4);
        std::memcpy(idbuf.buffer.contents,ids,rows*4); std::memcpy(maskbuf.buffer.contents,mask,rows*4);
        uint32_t M=static_cast<uint32_t>(rows),B=static_cast<uint32_t>(batch),T=static_cast<uint32_t>(length);
        NSMutableDictionary* descriptions=[NSMutableDictionary dictionary];
        Tensor x,ple,next_h;
        // Captures require CPU access at every layer. Normal inference keeps
        // the entire graph on the GPU, with ordered reusable scratch buffers.
        bool joined = directory.empty();
        id<MTLCommandBuffer> cb = joined ? [queue commandBuffer] : nil;
        id<MTLComputeCommandEncoder> enc = joined ? start(cb) : nil;
        @autoreleasepool {
            if (!joined) { cb=[queue commandBuffer]; enc=start(cb); }
            x=allocate(rows*hidden); float scale=rounded(std::sqrt(float(hidden)));
            ElementParams p{static_cast<uint32_t>(x.count),hidden,0,0,0,scale};
            element(enc,"embedding",{idbuf,weights.at("embed_tokens.weight"),x},p,x.count);
            if (ple_width) {
                Tensor projected=linear(enc,x,"ple.per_layer_model_projection",M);
                ple=norm(enc,projected,ple_width,"ple.per_layer_projection_norm.weight",{},std::pow(float(hidden),-0.5f));
            }
            if (!joined) finish(cb,enc);
            capture(directory,"embed",x,{batch,length,hidden},descriptions);
            if (ple_width) capture(directory,"ple",ple,{batch,length,layers.size(),ple_width},descriptions);
        }
        for(size_t i=0;i<layers.size();++i) @autoreleasepool {
            scratch_enabled = joined; scratch_cursor = 0;
            const auto& l=layers[i]; std::string p="layers."+std::to_string(i)+".";
            if (!joined) { cb=[queue commandBuffer]; enc=start(cb); }
            if (i == 0) next_h=norm(enc,x,hidden,p+"input_layernorm.weight");
            Tensor h=next_h, q, k, v;
            std::string flash = flash_kernel(l.dim, T);
            // The flash kernels read K transposed, which only the fused QKV
            // kernel produces.
            if (!weights.count(p+"self_attn.qkv_proj.weight")) flash.clear();
            if (weights.count(p+"self_attn.qkv_proj.weight")) {
                Tensor fused=linear(enc,h,p+"self_attn.qkv_proj",M);
                q=allocate(rows*heads*l.dim); k=allocate(rows*l.kv*l.dim); v=allocate(rows*l.kv*l.dim);
                QkvParams qp{M,heads,l.kv,l.dim,T,l.angles,!flash.empty(),eps,l.theta,l.factor,v_norm};
                dispatch(enc,"qkv_norm_rope",{fused,weights.at(p+"self_attn.q_norm.weight"),weights.at(p+"self_attn.k_norm.weight"),q,k,v},
                         qp,MTLSizeMake(heads+2*l.kv,M,1),MTLSizeMake(128,1,1));
            } else {
                Tensor q0=linear(enc,h,p+"self_attn.q_proj",M),k0=linear(enc,h,p+"self_attn.k_proj",M),v0=linear(enc,h,p+"self_attn.v_proj",M);
                Tensor qn=norm(enc,q0,l.dim,p+"self_attn.q_norm.weight"),kn=norm(enc,k0,l.dim,p+"self_attn.k_norm.weight");
                v=v_norm ? norm(enc,v0,l.dim,"") : v0;
                q=rotate(enc,qn,heads,T,l); k=rotate(enc,kn,l.kv,T,l);
            }
            AttnParams ap{B,T,heads,l.kv,l.dim,l.window};
            Tensor a0;
            if (!flash.empty()) {
                a0=allocate(rows*heads*l.dim);
                uint32_t r = flash[7]-'0', c = flash[9]-'0';  // "flash_r<R>c<C>d<ND>"
                dispatch(enc,flash,{q,k,v,maskbuf,a0},ap,MTLSizeMake((T+8*r-1)/(8*r),B*heads,1),MTLSizeMake(32*r*c,1,1));
            } else if (T >= 256 && l.dim <= 512) {
                a0=allocate(rows*heads*l.dim);
                dispatch(enc,"attention",{q,k,v,maskbuf,a0},ap,MTLSizeMake((T+7)/8,B*heads,1),MTLSizeMake(128,1,1));
            } else {
                Tensor scores=allocate(batch*heads*length*length,4),probs=allocate(scores.count);
                a0=allocate(rows*heads*l.dim);
                dispatch(enc,"qk",{q,k,maskbuf,scores},ap,MTLSizeMake((T+15)/16,(T+15)/16,B*heads),MTLSizeMake(16,16,1));
                dispatch(enc,"softmax",{scores,probs},T,MTLSizeMake(B*heads*T,1,1),MTLSizeMake(256,1,1));
                dispatch(enc,"pv",{probs,v,a0},ap,MTLSizeMake((l.dim+15)/16,(T+15)/16,B*heads),MTLSizeMake(16,16,1));
            }
            Tensor a=linear(enc,a0,p+"self_attn.o_proj",M);
            Tensor after_attn=allocate(a.count), mh=allocate(a.count);
            chain(enc,a,p+"post_attention_layernorm.weight",x,"",p+"pre_feedforward_layernorm.weight",after_attn,mh);
            Tensor gated;
            if (weights.count(p+"mlp.gateup_proj.weight")) gated=linear(enc,mh,p+"mlp.gateup_proj",M,1);
            else {
                Tensor gate=linear(enc,mh,p+"mlp.gate_proj",M),up=linear(enc,mh,p+"mlp.up_proj",M);
                gated=allocate(gate.count);
                ElementParams mp{static_cast<uint32_t>(gate.count),l.ffn,0,0,exact_gelu,1};
                element(enc,"gelu_mul",{gate,up,dummy,gated},mp,gate.count);
            }
            Tensor mlp=linear(enc,gated,p+"mlp.down_proj",M);
            int parity=int(i%2)*2;
            std::string next_norm=i+1<layers.size() ? "layers."+std::to_string(i+1)+".input_layernorm.weight" : "norm.weight";
            if (!ple_width) {
                // Gemma 3: the layer ends with the MLP residual; fuse the next input norm.
                Tensor out=carry(mlp.count,parity); next_h=carry(mlp.count,parity+1);
                chain(enc,mlp,p+"post_feedforward_layernorm.weight",after_attn,"",next_norm,out,next_h);
                if (!joined) finish(cb,enc);
                capture(directory,"layer"+std::to_string(i)+".attn",a,{batch,length,hidden},descriptions);
                capture(directory,"layer"+std::to_string(i)+".mlp",mlp,{batch,length,hidden},descriptions);
                capture(directory,"layer"+std::to_string(i)+".out",out,{batch,length,hidden},descriptions);
                x=out;
                continue;
            }
            Tensor after_mlp=allocate(mlp.count);
            chain(enc,mlp,p+"post_feedforward_layernorm.weight",after_attn,"","",after_mlp,{});
            std::string gate_name=p+"ple_block.per_layer_input_gate";
            Tensor pgated;
            if (transposed.count(gate_name+".weight")) pgated=linear(enc,after_mlp,gate_name,M,2,ple,static_cast<uint32_t>(i));
            else {
                Tensor pg=linear(enc,after_mlp,gate_name,M); pgated=allocate(pg.count);
                ElementParams pp{static_cast<uint32_t>(pg.count),ple_width,static_cast<uint32_t>(layers.size()),static_cast<uint32_t>(i),exact_gelu,1};
                element(enc,"gelu_mul",{pg,dummy,ple,pgated},pp,pg.count);
            }
            Tensor projected=linear(enc,pgated,p+"ple_block.per_layer_projection",M);
            // Residual, layer_scalar, and the next layer's input norm (or the
            // final norm after the last layer) in one pass.
            Tensor out=carry(projected.count,parity); next_h=carry(projected.count,parity+1);
            chain(enc,projected,p+"ple_block.post_per_layer_input_norm.weight",after_mlp,p+"layer_scalar",next_norm,out,next_h);
            if (!joined) finish(cb,enc);
            capture(directory,"layer"+std::to_string(i)+".attn",a,{batch,length,hidden},descriptions);
            capture(directory,"layer"+std::to_string(i)+".mlp",mlp,{batch,length,hidden},descriptions);
            capture(directory,"layer"+std::to_string(i)+".out",out,{batch,length,hidden},descriptions);
            x=out;
        }
        std::vector<float> result(batch*dim);
        scratch_enabled = false;
        @autoreleasepool {
            if (!joined) { cb=[queue commandBuffer]; enc=start(cb); }
            Tensor final=next_h,tokens=linear(enc,final,"embedding_projection",M);
            Tensor pooled=allocate(batch*output,4),out=allocate(batch*dim,4);
            PoolParams p{B,T,output,static_cast<uint32_t>(dim)};
            element(enc,"mean_pool",{tokens,maskbuf,pooled},p,pooled.count);
            dispatch(enc,"l2_normalize",{pooled,out},p,MTLSizeMake(B,1,1),MTLSizeMake(256,1,1));
            finish(cb,enc);
            capture(directory,"final_norm",final,{batch,length,hidden},descriptions);
            capture(directory,"embedding_projection",tokens,{batch,length,output},descriptions);
            std::memcpy(result.data(),out.buffer.contents,result.size()*4);
        }
        if(!directory.empty()) {
            std::ofstream f(std::filesystem::path(directory)/"captures.json"); f<<json_string(descriptions)<<'\n';
            require(f.good(),"Cannot write capture metadata");
        }
        require(std::all_of(result.begin(),result.end(),[](float v){return std::isfinite(v);}),"Non-finite embedding output");
        NSMutableDictionary* counts = [NSMutableDictionary dictionary];
        NSMutableDictionary* times = [NSMutableDictionary dictionary];
        for (const auto& item : dispatch_counts) counts[ns(item.first)] = @(item.second);
        if (profiling) {
            MTLTimestamp cpu_end, gpu_end;
            [device sampleTimestamps:&cpu_end gpuTimestamp:&gpu_end];
            require(gpu_end > gpu_start && cpu_end > cpu_start, "Invalid timestamp calibration");
            double ns_per_tick = double(cpu_end-cpu_start)/double(gpu_end-gpu_start);
            NSData* data = [samples resolveCounterRange:NSMakeRange(0, 2*dispatch_names.size())];
            require(data.length == 2*dispatch_names.size()*sizeof(MTLCounterResultTimestamp), "Invalid GPU timestamp result size");
            const auto* values = static_cast<const MTLCounterResultTimestamp*>(data.bytes);
            for (size_t i = 0; i < dispatch_names.size(); ++i) {
                uint64_t begin = values[2*i].timestamp, end = values[2*i+1].timestamp;
                require(begin != MTLCounterErrorValue && end != MTLCounterErrorValue && end >= begin, "Invalid GPU timestamp sample");
                NSString* name = ns(dispatch_names[i]);
                times[name] = @([times[name] doubleValue] + (end-begin)*ns_per_tick*1e-6);
            }
        }
        last_profile = json_string(@{@"dispatch_counts":counts, @"dispatches":@(dispatch_names.size()),
            @"cpu_gpu_waits":@(waits), @"gpu_buffer_barriers":@(dispatch_names.size()),
            @"command_buffer_gpu_ms":@(gpu_seconds*1e3), @"cpu_wait_ms":@(wait_seconds*1e3),
            @"peak_metal_allocated_bytes":@(peak_allocated), @"kernel_gpu_ms":times,
            @"timestamp_mode":profiling ? (dispatch_sampling ? @"dispatch/barrier" : @"one encoder per dispatch") : @"disabled"});
        pad_from = pad_to = 0;
        return result;
    }
};

Engine::Engine(const std::string& path,const std::string& dtype,bool profile,const std::string& matmul) {
    @autoreleasepool { impl_=std::make_unique<Impl>(path,dtype,profile,matmul); }
}
Engine::~Engine()=default;
std::vector<float> Engine::encode(const int32_t* ids,const int32_t* mask,size_t b,size_t t,size_t dim,const std::string& dir) {
    @autoreleasepool { return impl_->encode(ids,mask,b,t,dim,dir); }
}
std::string Engine::info() const { @autoreleasepool { return impl_->info(); } }
std::string Engine::profile() const { std::lock_guard<std::mutex> lock(impl_->mutex); return impl_->last_profile; }
}
