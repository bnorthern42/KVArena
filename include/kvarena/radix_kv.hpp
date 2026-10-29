#pragma once

#include "kvarena/arena.hpp"
#include <cstdint>

namespace kvarena {

inline void insert_radix_entry(SharedArena& arena, const RadixBlockEntry& entry) {
    auto* table = arena.header()->radix_table;
    RadixBlockEntry current = entry;
    uint32_t idx = current.block_hash % MAX_INDEX_ENTRIES;

    for (uint32_t probe = 0; probe < MAX_INDEX_ENTRIES; ++probe) {
        if (!table[idx].is_valid || table[idx].is_tombstone) {
            table[idx] = current;
            return;
        }

        if (table[idx].block_hash == current.block_hash) {
            // Update existing
            table[idx] = current;
            return;
        }

        if (current.probe_dist > table[idx].probe_dist) {
            // Robin Hood swap
            std::swap(current, table[idx]);
        }

        current.probe_dist++;
        idx = (idx + 1) % MAX_INDEX_ENTRIES;
    }
}

inline bool lookup_radix_entry(SharedArena& arena, uint64_t block_hash, RadixBlockEntry& out_entry) {
    auto* table = arena.header()->radix_table;
    uint32_t idx = block_hash % MAX_INDEX_ENTRIES;

    for (uint32_t probe = 0; probe < MAX_INDEX_ENTRIES; ++probe) {
        if (!table[idx].is_valid) {
            return false; // Empty slot, stop probing
        }

        if (!table[idx].is_tombstone && table[idx].block_hash == block_hash) {
            out_entry = table[idx];
            return true;
        }

        if (probe > table[idx].probe_dist) {
            return false; // Found an element with lower probe distance, we should have seen ours by now
        }

        idx = (idx + 1) % MAX_INDEX_ENTRIES;
    }
    return false;
}

} // namespace kvarena
