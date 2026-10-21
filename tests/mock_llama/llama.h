#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>

extern "C" {

typedef int32_t llama_token;
typedef int32_t llama_pos;
typedef int32_t llama_seq_id;

struct llama_vocab {
    int dummy;
};

struct llama_model {
    llama_vocab vocab;
};

// Each simulated KV cell represents 64 bytes of K/V tensor data for one (token, pos)
struct mock_kv_cell {
    llama_token token;
    llama_pos   pos;
    uint64_t    rope_rotated_signature;
    uint8_t     tensor_payload[48]; // Pads cell to exactly 64 bytes

    static uint64_t compute_sig(llama_token t, llama_pos p) {
        uint64_t h = 14695981039346656037ULL;
        h = (h ^ static_cast<uint64_t>(t)) * 1099511628211ULL;
        h = (h ^ static_cast<uint64_t>(p)) * 1099511628211ULL;
        return h;
    }

    void populate(llama_token t, llama_pos p) {
        token = t;
        pos = p;
        rope_rotated_signature = compute_sig(t, p);
        for (size_t i = 0; i < sizeof(tensor_payload); ++i) {
            tensor_payload[i] = static_cast<uint8_t>((rope_rotated_signature >> ((i % 8) * 8)) ^ i);
        }
    }
};
static_assert(sizeof(mock_kv_cell) == 64, "mock_kv_cell must be 64 bytes");

struct llama_memory_t {
    // Maps seq_id -> ordered map of (pos -> mock_kv_cell)
    std::map<llama_seq_id, std::map<llama_pos, mock_kv_cell>> sequences;
};

typedef llama_memory_t* llama_memory;

struct llama_context {
    const llama_model* model = nullptr;
    llama_memory_t     mem;
};

struct llama_batch {
    int32_t        n_tokens;
    int32_t        n_tokens_alloc; // Tracks allocation size for clean teardown
    llama_token*   token;
    float*         embd;
    llama_pos*     pos;
    int32_t*       n_seq_id;
    llama_seq_id** seq_id;
    int8_t*        logits;
};

// ----------------------------------------------------------------------------
// Lifecycle & Vocab Stubs
// ----------------------------------------------------------------------------
inline void llama_backend_init() {}
inline void llama_backend_free() {}

inline llama_model* llama_model_load_from_file(const char*, struct llama_model_params*) {
    return new llama_model{};
}
inline void llama_model_free(llama_model* model) { delete model; }

inline const llama_vocab* llama_model_get_vocab(const llama_model* model) {
    return &model->vocab;
}

inline llama_context* llama_init_from_model(llama_model* model, struct llama_context_params*) {
    auto* ctx = new llama_context{};
    ctx->model = model;
    return ctx;
}
inline void llama_free(llama_context* ctx) { delete ctx; }

inline llama_memory llama_get_memory(llama_context* ctx) {
    return &ctx->mem;
}

// Byte-level deterministic mock tokenizer (1 char = 1 token)
inline int32_t llama_tokenize(
    const struct llama_vocab*,
    const char* text,
    int32_t text_len,
    llama_token* tokens,
    int32_t n_tokens_max,
    bool add_special,
    bool /*parse_special*/)
{
    int32_t needed = text_len + (add_special ? 1 : 0);
    if (tokens == nullptr || n_tokens_max < needed) {
        return -needed;
    }
    int32_t idx = 0;
    if (add_special) {
        tokens[idx++] = 1; // BOS token
    }
    for (int32_t i = 0; i < text_len; ++i) {
        tokens[idx++] = static_cast<llama_token>(static_cast<uint8_t>(text[i])) + 100;
    }
    return idx;
}

// ----------------------------------------------------------------------------
// Batch & Decode Simulation
// ----------------------------------------------------------------------------
inline llama_batch llama_batch_init(int32_t n_tokens_alloc, int32_t /*embd*/, int32_t n_seq_max) {
    llama_batch b{};
    b.n_tokens = 0;
    b.n_tokens_alloc = n_tokens_alloc;
    b.token    = new llama_token[n_tokens_alloc]();
    b.embd     = nullptr;
    b.pos      = new llama_pos[n_tokens_alloc]();
    b.n_seq_id = new int32_t[n_tokens_alloc]();
    b.seq_id   = new llama_seq_id*[n_tokens_alloc]();
    for (int32_t i = 0; i < n_tokens_alloc; ++i) {
        b.seq_id[i] = new llama_seq_id[n_seq_max]();
    }
    b.logits   = new int8_t[n_tokens_alloc]();
    return b;
}

inline void llama_batch_free(llama_batch b) {
    delete[] b.token;
    delete[] b.pos;
    delete[] b.n_seq_id;
    if (b.seq_id) {
        for (int32_t i = 0; i < b.n_tokens_alloc; ++i) {
            delete[] b.seq_id[i];
        }
    }
    delete[] b.seq_id;
    delete[] b.logits;
}

inline int32_t llama_decode(llama_context* ctx, llama_batch batch) {
    if (!ctx || batch.n_tokens <= 0) return -1;
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        llama_token t = batch.token[i];
        llama_pos   p = batch.pos[i];
        for (int32_t s = 0; s < batch.n_seq_id[i]; ++s) {
            llama_seq_id seq = batch.seq_id[i][s];
            mock_kv_cell cell{};
            cell.populate(t, p);
            ctx->mem.sequences[seq][p] = cell;
        }
    }
    return 0;
}

// ----------------------------------------------------------------------------
// Sequence KV State Serialization & Alignment Verification
// ----------------------------------------------------------------------------
inline size_t llama_state_seq_get_size(llama_context* ctx, llama_seq_id seq_id) {
    auto it = ctx->mem.sequences.find(seq_id);
    if (it == ctx->mem.sequences.end() || it->second.empty()) return 0;
    // 64-byte header + 64 bytes per KV cell
    return 64 + it->second.size() * sizeof(mock_kv_cell);
}

inline size_t llama_state_seq_get_data(llama_context* ctx, uint8_t* dst, size_t size, llama_seq_id seq_id) {
    // Enforce AGENTS.md Invariant #3: 64-byte alignment for KV state buffers
    if (reinterpret_cast<uintptr_t>(dst) % 64 != 0) {
        throw std::runtime_error("AGENTS.md violation: KV destination buffer is not 64-byte aligned!");
    }
    size_t needed = llama_state_seq_get_size(ctx, seq_id);
    if (needed == 0 || size < needed) return 0;

    std::memset(dst, 0, 64);
    uint64_t count = ctx->mem.sequences[seq_id].size();
    std::memcpy(dst, &count, sizeof(uint64_t));

    uint8_t* cell_dst = dst + 64;
    for (const auto& [pos, cell] : ctx->mem.sequences[seq_id]) {
        std::memcpy(cell_dst, &cell, sizeof(mock_kv_cell));
        cell_dst += sizeof(mock_kv_cell);
    }
    return needed;
}

inline size_t llama_state_seq_set_data(llama_context* ctx, const uint8_t* src, size_t size, llama_seq_id dest_seq_id) {
    // Enforce AGENTS.md Invariant #3: 64-byte alignment for KV state buffers
    if (reinterpret_cast<uintptr_t>(src) % 64 != 0) {
        throw std::runtime_error("AGENTS.md violation: KV source buffer is not 64-byte aligned!");
    }
    if (size < 64) return 0;

    uint64_t count = 0;
    std::memcpy(&count, src, sizeof(uint64_t));
    size_t expected_size = 64 + count * sizeof(mock_kv_cell);
    if (size < expected_size) return 0;

    auto& seq_map = ctx->mem.sequences[dest_seq_id];
    seq_map.clear();

    const uint8_t* cell_src = src + 64;
    for (uint64_t i = 0; i < count; ++i) {
        mock_kv_cell cell{};
        std::memcpy(&cell, cell_src, sizeof(mock_kv_cell));
        seq_map[cell.pos] = cell;
        cell_src += sizeof(mock_kv_cell);
    }
    return expected_size;
}

// ----------------------------------------------------------------------------
// Phase 3 Support: RoPE In-Place Position Shifting
// ----------------------------------------------------------------------------
inline void llama_memory_seq_add(llama_memory mem, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta) {
    auto it = mem->sequences.find(seq_id);
    if (it == mem->sequences.end()) return;

    std::map<llama_pos, mock_kv_cell> updated;
    for (auto& [pos, cell] : it->second) {
        bool in_range = (pos >= p0) && (p1 < 0 || pos < p1);
        if (in_range) {
            llama_pos new_pos = pos + delta;
            cell.populate(cell.token, new_pos); // Re-rotates synthetic RoPE signature
            updated[new_pos] = cell;
        } else {
            updated[pos] = cell;
        }
    }
    it->second = std::move(updated);
}

} // extern "C"
