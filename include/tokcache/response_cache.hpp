#pragma once

#include "tokcache/arena.hpp"
#include "llama.h"
#include <span>
#include <vector>

namespace tokcache {

inline void insert_response_entry(SharedArena& arena, const ResponseCacheEntry& entry) {
    auto* table = arena.header()->resp_table;
    ResponseCacheEntry current = entry;
    uint32_t idx = current.query_hash % MAX_RESP_ENTRIES;

    for (uint32_t probe = 0; probe < MAX_RESP_ENTRIES; ++probe) {
        if (!table[idx].is_valid || table[idx].is_tombstone) {
            table[idx] = current;
            return;
        }

        if (table[idx].query_hash == current.query_hash) {
            // Update existing
            table[idx] = current;
            return;
        }

        if (current.probe_dist > table[idx].probe_dist) {
            // Robin Hood swap
            std::swap(current, table[idx]);
        }

        current.probe_dist++;
        idx = (idx + 1) % MAX_RESP_ENTRIES;
    }
}

inline bool lookup_response_cache(SharedArena& arena, uint64_t model_hash, std::span<const llama_token> prompt, uint64_t sampling_hash, std::vector<llama_token>& out) {
    uint64_t q_hash = hash_bytes(prompt.data(), prompt.size_bytes(), model_hash ^ sampling_hash);
    uint32_t idx = q_hash % MAX_RESP_ENTRIES;
    auto* table = arena.header()->resp_table;

    arena.lock_read();

    bool found = false;
    for (uint32_t probe = 0; probe < MAX_RESP_ENTRIES; ++probe) {
        if (!table[idx].is_valid) {
            break; // Empty slot
        }

        if (!table[idx].is_tombstone && table[idx].query_hash == q_hash) {
            if (table[idx].token_offset != 0) {
                const auto* ptr = arena.ptr_from_offset<const llama_token>(table[idx].token_offset);
                out.assign(ptr, ptr + table[idx].n_output_tokens);
                found = true;
            }
            break;
        }

        if (probe > table[idx].probe_dist) {
            break;
        }

        idx = (idx + 1) % MAX_RESP_ENTRIES;
    }

    arena.unlock();
    return found;
}

inline void store_response_cache(SharedArena& arena, uint64_t model_hash, std::span<const llama_token> prompt, uint64_t sampling_hash, std::span<const llama_token> response) {
    uint64_t q_hash = hash_bytes(prompt.data(), prompt.size_bytes(), model_hash ^ sampling_hash);

    arena.lock_write();
    uint64_t offset = arena.allocate(response.size_bytes(), alignof(llama_token));
    std::memcpy(arena.ptr_from_offset<llama_token>(offset), response.data(), response.size_bytes());

    ResponseCacheEntry entry = {
        .query_hash = q_hash,
        .token_offset = offset,
        .n_output_tokens = static_cast<uint32_t>(response.size()),
        .probe_dist = 0,
        .is_valid = true,
        .is_tombstone = false
    };

    insert_response_entry(arena, entry);

    arena.unlock();
}

} // namespace tokcache
