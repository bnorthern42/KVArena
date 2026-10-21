#pragma once

#include "llama.h"
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <pthread.h>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace tokcache {

constexpr size_t BLOCK_TOKENS = 64;          // Granularity for prefix matching
constexpr size_t MAX_INDEX_ENTRIES = 4096;   // Fixed-size table in SHM
constexpr size_t MAX_RESP_ENTRIES = 1024;    // Deterministic response cache entries
constexpr uint64_t MAGIC_HEADER = 0x544F4B4341434845ULL; // "TOKCACHE"

// Fast 64-bit FNV-1a rolling hash for token blocks and sampling params
inline uint64_t hash_bytes(const void* data, size_t len, uint64_t seed = 14695981039346656037ULL) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    uint64_t h = seed;
    for (size_t i = 0; i < len; ++i) {
        h ^= bytes[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// ============================================================================
// LAYER B: TokenRef (Lightweight IPC descriptor passed between agents)
// ============================================================================
struct TokenRef {
    uint64_t model_hash;      // Ensures agents use compatible vocab/weights
    uint64_t prefix_hash;     // Hash of the longest block-aligned prefix
    uint64_t token_offset;    // Byte offset in SHM data slab to raw llama_token[]
    uint32_t n_tokens;        // Number of tokens in this span
    uint64_t kv_offset;       // Byte offset in SHM data slab to serialized KV state (0 if none)
    uint64_t kv_size;         // Size of serialized KV state in bytes
};

// ============================================================================
// LAYER A & C: Shared Memory Data Structures (Offset-Based, No Raw Pointers)
// ============================================================================
struct RadixBlockEntry {
    uint64_t block_hash;      // Chained hash: H(parent_hash + model_hash + 64 tokens)
    uint64_t parent_hash;     // Hash of preceding block (0 for root)
    uint64_t token_offset;    // Offset in slab where full prefix tokens are stored
    uint32_t total_tokens;    // Total tokens from pos 0 up to the end of this block
    uint64_t kv_offset;       // Offset in slab where llama_state_seq_get_data is stored
    uint64_t kv_size;         // Byte size of the KV state snapshot
    uint64_t last_access_ts;  // For LRU eviction
};

struct ResponseCacheEntry {
    uint64_t query_hash;      // H(model_hash + full_prompt_tokens + sampling_params)
    uint64_t token_offset;    // Offset in slab to generated output llama_token[]
    uint32_t n_output_tokens; // Length of cached output token sequence
};

struct ShmHeader {
    uint64_t magic;
    pthread_rwlock_t rwlock;  // Cross-process read-write lock
    uint64_t slab_capacity;
    uint64_t slab_head;       // Bump allocator offset inside the data slab

    RadixBlockEntry radix_table[MAX_INDEX_ENTRIES];
    ResponseCacheEntry resp_table[MAX_RESP_ENTRIES];
    // Immediately following ShmHeader in memory is the raw byte slab: uint8_t slab[]
};

// ============================================================================
// Shared Memory Arena Manager
// ============================================================================
class SharedArena {
public:
    SharedArena(const std::string& shm_name, size_t total_bytes, bool create_or_reset = false)
        : shm_name_(shm_name), total_bytes_(total_bytes) {
        
        int flags = O_RDWR | O_CREAT;
        fd_ = shm_open(shm_name_.c_str(), flags, 0666);
        if (fd_ < 0) throw std::runtime_error("Failed to shm_open: " + shm_name_);

        struct stat st{};
        fstat(fd_, &st);
        bool is_new = (st.st_size == 0) || create_or_reset;

        if (is_new) {
            if (ftruncate(fd_, static_cast<off_t>(total_bytes_)) != 0) {
                close(fd_);
                throw std::runtime_error("Failed to ftruncate shared memory");
            }
        }

        base_ = static_cast<uint8_t*>(
            mmap(nullptr, total_bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0)
        );
        if (base_ == MAP_FAILED) {
            close(fd_);
            throw std::runtime_error("Failed to mmap shared memory");
        }

        header_ = reinterpret_cast<ShmHeader*>(base_);
        size_t header_size = sizeof(ShmHeader);
        size_t padding = (64 - (header_size % 64)) % 64;
        slab_base_ = base_ + header_size + padding;

        if (is_new || header_->magic != MAGIC_HEADER) {
            std::memset(header_, 0, sizeof(ShmHeader));
            pthread_rwlockattr_t attr;
            pthread_rwlockattr_init(&attr);
            pthread_rwlockattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
            pthread_rwlock_init(&header_->rwlock, &attr);
            pthread_rwlockattr_destroy(&attr);

            header_->slab_capacity = total_bytes_ - (slab_base_ - base_);
            header_->slab_head = 8; // Reserve offset 0 as "null"
            header_->magic = MAGIC_HEADER;
        }
    }

    ~SharedArena() {
        if (base_ && base_ != MAP_FAILED) munmap(base_, total_bytes_);
        if (fd_ >= 0) close(fd_);
    }

    // Lock helpers
    void lock_read()  { pthread_rwlock_rdlock(&header_->rwlock); }
    void lock_write() { pthread_rwlock_wrlock(&header_->rwlock); }
    void unlock()     { pthread_rwlock_unlock(&header_->rwlock); }

    // Allocate aligned bytes in the shared slab (caller must hold write lock)
    uint64_t allocate(size_t bytes, size_t alignment = 8) {
        uint64_t aligned_head = (header_->slab_head + (alignment - 1)) & ~(alignment - 1);
        if (aligned_head + bytes > header_->slab_capacity) {
            // Simple wrap-around reset for demo; replace with slab/LRU freelist in production
            aligned_head = 8;
            std::memset(header_->radix_table, 0, sizeof(header_->radix_table));
            std::memset(header_->resp_table, 0, sizeof(header_->resp_table));
        }
        header_->slab_head = aligned_head + bytes;
        return aligned_head;
    }

    template <typename T>
    T* ptr_from_offset(uint64_t offset) const {
        if (offset == 0) return nullptr;
        return reinterpret_cast<T*>(slab_base_ + offset);
    }

    ShmHeader* header() const { return header_; }

private:
    std::string shm_name_;
    size_t total_bytes_;
    int fd_ = -1;
    uint8_t* base_ = nullptr;
    ShmHeader* header_ = nullptr;
    uint8_t* slab_base_ = nullptr;
};

// ============================================================================
// Unified Agent Cache Client (Integrates Layers A, B, and C with llama.cpp)
// ============================================================================
class AgentTokenCache {
public:
    AgentTokenCache(SharedArena& arena, llama_model* model, llama_context* ctx, std::string_view model_id)
        : arena_(arena), model_(model), ctx_(ctx) {
        model_hash_ = hash_bytes(model_id.data(), model_id.size());
    }

    // ------------------------------------------------------------------------
    // LAYER B: Zero-Copy Token Sharing (Write tokens to SHM, return TokenRef)
    // ------------------------------------------------------------------------
    TokenRef publish_tokens(std::span<const llama_token> tokens) {
        arena_.lock_write();
        uint64_t tok_offset = arena_.allocate(tokens.size_bytes(), alignof(llama_token));
        std::memcpy(arena_.ptr_from_offset<llama_token>(tok_offset), tokens.data(), tokens.size_bytes());
        arena_.unlock();

        return TokenRef{
            .model_hash = model_hash_,
            .prefix_hash = compute_chain_hash(tokens),
            .token_offset = tok_offset,
            .n_tokens = static_cast<uint32_t>(tokens.size()),
            .kv_offset = 0,
            .kv_size = 0
        };
    }

    // Zero-copy view directly into shared memory token array
    std::span<const llama_token> resolve_tokens(const TokenRef& ref) const {
        if (ref.model_hash != model_hash_) {
            throw std::runtime_error("Model mismatch on TokenRef resolution");
        }
        const auto* ptr = arena_.ptr_from_offset<const llama_token>(ref.token_offset);
        return std::span<const llama_token>(ptr, ref.n_tokens);
    }

    // ------------------------------------------------------------------------
    // LAYER C: Deterministic Response Cache Lookup & Store
    // ------------------------------------------------------------------------
    bool lookup_response(std::span<const llama_token> prompt, uint64_t sampling_hash, std::vector<llama_token>& out) {
        uint64_t q_hash = hash_bytes(prompt.data(), prompt.size_bytes(), model_hash_ ^ sampling_hash);
        size_t idx = q_hash % MAX_RESP_ENTRIES;

        arena_.lock_read();
        const auto& entry = arena_.header()->resp_table[idx];
        if (entry.query_hash == q_hash && entry.token_offset != 0) {
            const auto* ptr = arena_.ptr_from_offset<const llama_token>(entry.token_offset);
            out.assign(ptr, ptr + entry.n_output_tokens);
            arena_.unlock();
            return true;
        }
        arena_.unlock();
        return false;
    }

    void store_response(std::span<const llama_token> prompt, uint64_t sampling_hash, std::span<const llama_token> response) {
        uint64_t q_hash = hash_bytes(prompt.data(), prompt.size_bytes(), model_hash_ ^ sampling_hash);
        size_t idx = q_hash % MAX_RESP_ENTRIES;

        arena_.lock_write();
        uint64_t offset = arena_.allocate(response.size_bytes(), alignof(llama_token));
        std::memcpy(arena_.ptr_from_offset<llama_token>(offset), response.data(), response.size_bytes());

        arena_.header()->resp_table[idx] = ResponseCacheEntry{
            .query_hash = q_hash,
            .token_offset = offset,
            .n_output_tokens = static_cast<uint32_t>(response.size())
        };
        arena_.unlock();
    }

    // ------------------------------------------------------------------------
    // LAYER A: Radix Prefix KV-Cache Lookup + Delta Prefill + Snapshot Save
    // ------------------------------------------------------------------------
    void prefill_with_kv_cache(std::span<const llama_token> tokens, llama_seq_id seq_id = 0) {
        size_t num_blocks = tokens.size() / BLOCK_TOKENS;

        // 1. Walk block chain to find the deepest cached KV prefix in SHM
        uint64_t running_hash = model_hash_;
        std::vector<uint64_t> block_hashes(num_blocks);
        for (size_t b = 0; b < num_blocks; ++b) {
            auto block_span = tokens.subspan(b * BLOCK_TOKENS, BLOCK_TOKENS);
            running_hash = hash_bytes(block_span.data(), block_span.size_bytes(), running_hash);
            block_hashes[b] = running_hash;
        }

        size_t matched_tokens = 0;
        arena_.lock_read();
        for (ssize_t b = static_cast<ssize_t>(num_blocks) - 1; b >= 0; --b) {
            uint64_t h = block_hashes[b];
            const auto& entry = arena_.header()->radix_table[h % MAX_INDEX_ENTRIES];
            if (entry.block_hash == h && entry.kv_offset != 0) {
                const auto* kv_ptr = arena_.ptr_from_offset<const uint8_t>(entry.kv_offset);
                // Inject cached KV state directly from mmap buffer into llama_context
                size_t loaded = llama_state_seq_set_data(ctx_, kv_ptr, entry.kv_size, seq_id);
                if (loaded == entry.kv_size) {
                    matched_tokens = entry.total_tokens;
                    auto it = ctx_->mem.sequences.find(seq_id);
                    if (it != ctx_->mem.sequences.end()) {
                        for (auto pos_it = it->second.begin(); pos_it != it->second.end(); ) {
                            if (pos_it->first >= static_cast<llama_pos>(matched_tokens)) pos_it = it->second.erase(pos_it);
                            else ++pos_it;
                        }
                    }
                    break;
                }
            }
        }
        arena_.unlock();

        // 2. Compute only the uncached suffix tokens
        if (matched_tokens < tokens.size()) {
            // If we always want the last token's logits ready for sampling, ensure at least 1 token is decoded
            // if (matched_tokens == tokens.size()) matched_tokens -= 1; // Removed for strict byte-for-byte check

            auto suffix = tokens.subspan(matched_tokens);
            decode_span(suffix, static_cast<llama_pos>(matched_tokens), seq_id);

            // 3. If we crossed a block boundary, snapshot the new prefix KV state into SHM
            if (num_blocks > 0 && matched_tokens < num_blocks * BLOCK_TOKENS) {
                save_kv_block_checkpoint(tokens.first(num_blocks * BLOCK_TOKENS),
                                         block_hashes.back(),
                                         num_blocks > 1 ? block_hashes[num_blocks - 2] : 0,
                                         seq_id);
            }
        }
    }

private:
    uint64_t compute_chain_hash(std::span<const llama_token> tokens) const {
        return hash_bytes(tokens.data(), tokens.size_bytes(), model_hash_);
    }

    void decode_span(std::span<const llama_token> suffix, llama_pos start_pos, llama_seq_id seq_id) {
        if (suffix.empty()) return;
        llama_batch batch = llama_batch_init(static_cast<int32_t>(suffix.size()), 0, 1);
        for (size_t i = 0; i < suffix.size(); ++i) {
            batch.token[i]     = suffix[i];
            batch.pos[i]       = start_pos + static_cast<llama_pos>(i);
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = seq_id;
            batch.logits[i]    = (i == suffix.size() - 1); // Only compute logits for the final token
        }
        batch.n_tokens = static_cast<int32_t>(suffix.size());

        if (llama_decode(ctx_, batch) != 0) {
            llama_batch_free(batch);
            throw std::runtime_error("llama_decode failed during suffix prefill");
        }
        llama_batch_free(batch);
    }

    void save_kv_block_checkpoint(std::span<const llama_token> prefix_tokens,
                                  uint64_t block_hash,
                                  uint64_t parent_hash,
                                  llama_seq_id seq_id) {
        size_t kv_size = llama_state_seq_get_size(ctx_, seq_id);
        if (kv_size == 0) return;

        arena_.lock_write();
        uint64_t tok_offset = arena_.allocate(prefix_tokens.size_bytes(), alignof(llama_token));
        uint64_t kv_offset  = arena_.allocate(kv_size, 64); // 64-byte cacheline / SIMD aligned

        std::memcpy(arena_.ptr_from_offset<llama_token>(tok_offset),
                    prefix_tokens.data(),
                    prefix_tokens.size_bytes());

        auto* kv_dst = arena_.ptr_from_offset<uint8_t>(kv_offset);
        llama_state_seq_get_data(ctx_, kv_dst, kv_size, seq_id);

        size_t idx = block_hash % MAX_INDEX_ENTRIES;
        arena_.header()->radix_table[idx] = RadixBlockEntry{
            .block_hash     = block_hash,
            .parent_hash    = parent_hash,
            .token_offset   = tok_offset,
            .total_tokens   = static_cast<uint32_t>(prefix_tokens.size()),
            .kv_offset      = kv_offset,
            .kv_size        = kv_size,
            .last_access_ts = 0
        };
        arena_.unlock();
    }

    SharedArena& arena_;
    llama_model* model_;
    llama_context* ctx_;
    uint64_t model_hash_;
};

} // namespace tokcache
