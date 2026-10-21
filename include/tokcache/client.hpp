#pragma once

#include "tokcache/arena.hpp"
#include "tokcache/radix_kv.hpp"
#include "tokcache/response_cache.hpp"

#include "llama.h"
#include <span>
#include <vector>
#include <stdexcept>
#include <string_view>
#include <atomic>

namespace tokcache {

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

        // Custom allocate to get header offset
        size_t alignment = alignof(llama_token);
        uint64_t aligned_data_start = arena_.allocate(tokens.size_bytes(), alignment);

        // Find header by physical block walk - naive but works since we just allocated it and it's the most recent LRU head
        uint64_t header_offset = arena_.header()->lru_head;

        std::memcpy(arena_.ptr_from_offset<llama_token>(aligned_data_start), tokens.data(), tokens.size_bytes());
        arena_.unlock();

        return TokenRef{
            .model_hash = model_hash_,
            .prefix_hash = compute_chain_hash(tokens),
            .token_offset = aligned_data_start,
            .n_tokens = static_cast<uint32_t>(tokens.size()),
            .kv_offset = 0,
            .kv_size = 0,
            .token_header_offset = header_offset,
            .kv_header_offset = 0
        };
    }

    void pin(const TokenRef& ref) {
        if (ref.token_header_offset != 0) {
            auto* hdr = arena_.ptr_from_offset<AllocHeader>(ref.token_header_offset);
            std::atomic_ref<uint64_t> count(hdr->ref_count);
            count.fetch_add(1);
        }
        if (ref.kv_header_offset != 0) {
            auto* hdr = arena_.ptr_from_offset<AllocHeader>(ref.kv_header_offset);
            std::atomic_ref<uint64_t> count(hdr->ref_count);
            count.fetch_add(1);
        }
    }

    void unpin(const TokenRef& ref) {
        if (ref.token_header_offset != 0) {
            auto* hdr = arena_.ptr_from_offset<AllocHeader>(ref.token_header_offset);
            std::atomic_ref<uint64_t> count(hdr->ref_count);
            count.fetch_sub(1);
        }
        if (ref.kv_header_offset != 0) {
            auto* hdr = arena_.ptr_from_offset<AllocHeader>(ref.kv_header_offset);
            std::atomic_ref<uint64_t> count(hdr->ref_count);
            count.fetch_sub(1);
        }
    }

    // Zero-copy view directly into shared memory token array
    std::span<const llama_token> resolve_tokens(const TokenRef& ref) const {
        if (ref.model_hash != model_hash_) {
            throw std::runtime_error("Model mismatch on TokenRef resolution");
        }
        // Update access timestamp
        if (ref.token_header_offset != 0) {
            arena_.ptr_from_offset<AllocHeader>(ref.token_header_offset)->last_access_ts = get_monotonic_ns();
        }
        const auto* ptr = arena_.ptr_from_offset<const llama_token>(ref.token_offset);
        return std::span<const llama_token>(ptr, ref.n_tokens);
    }

    // ------------------------------------------------------------------------
    // LAYER C: Deterministic Response Cache Lookup & Store
    // ------------------------------------------------------------------------
    bool lookup_response(std::span<const llama_token> prompt, uint64_t sampling_hash, std::vector<llama_token>& out) {
        return lookup_response_cache(arena_, model_hash_, prompt, sampling_hash, out);
    }

    void store_response(std::span<const llama_token> prompt, uint64_t sampling_hash, std::span<const llama_token> response) {
        store_response_cache(arena_, model_hash_, prompt, sampling_hash, response);
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

            RadixBlockEntry entry;
            if (lookup_radix_entry(arena_, h, entry)) {
                const auto* kv_ptr = arena_.ptr_from_offset<const uint8_t>(entry.kv_offset);

                // We'd ideally update access ts, but we read-locked. Robin hood doesn't easily allow in-place modification if it moves.
                // We'll skip precise LRU access update here for simplicity, or we could write lock for access.

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

        RadixBlockEntry entry = {
            .block_hash     = block_hash,
            .parent_hash    = parent_hash,
            .token_offset   = tok_offset,
            .total_tokens   = static_cast<uint32_t>(prefix_tokens.size()),
            .kv_offset      = kv_offset,
            .kv_size        = kv_size,
            .last_access_ts = get_monotonic_ns(),
            .probe_dist     = 0,
            .is_valid       = true,
            .is_tombstone   = false
        };

        insert_radix_entry(arena_, entry);
        arena_.unlock();
    }

    SharedArena& arena_;
    llama_model* model_;
    llama_context* ctx_;
    uint64_t model_hash_;
};

} // namespace tokcache
