#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace egemma {
// Thread-safe per engine: encode calls are serialized around the Metal queue.
class Engine {
public:
    // matmul: "native" (own SIMD-group kernels only), "mpp" (Metal Performance
    // Primitives matmul2d for large enough projections; error if unavailable),
    // or "auto" (mpp on Apple10-family GPUs where it compiles, else native).
    // With "auto", the EGEMMA_MATMUL environment variable may override.
    Engine(const std::string& model_dir, const std::string& dtype = "fp32", bool profile = false,
           const std::string& matmul = "auto");
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    std::vector<float> encode(const int32_t* ids, const int32_t* mask,
                              size_t batch, size_t length, size_t dim = 768,
                              const std::string& capture_dir = "");
    std::string info() const;
    std::string profile() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
std::string device_info();
}
