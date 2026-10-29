#include "kvarena.hpp"
#include <iostream>
#include <vector>

// Helper to tokenize a string using llama.cpp's vocab API
std::vector<llama_token> tokenize(const llama_model* model, std::string_view text, bool add_special) {
    const llama_vocab* vocab = llama_model_get_vocab(model);
    int n_tokens = -llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, add_special, true);
    std::vector<llama_token> tokens(n_tokens);
    llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), add_special, true);
    return tokens;
}

int main() {
    // 1. Initialize shared memory segment (e.g., 512 MB for KV states & token arrays)
    kvarena::SharedArena shm("/kvarena_bus", 512ULL * 1024 * 1024, true);

    // 2. Load llama.cpp model & create separate contexts for Agent A and Agent B
    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    llama_model* model = llama_model_load_from_file("model.gguf", mparams);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 4096;
    llama_context* ctx_agent_a = llama_init_from_model(model, cparams);
    llama_context* ctx_agent_b = llama_init_from_model(model, cparams);

    kvarena::AgentTokenCache cache_a(shm, model, ctx_agent_a, "qwen2.5-coder-7b-q4_k_m");
    kvarena::AgentTokenCache cache_b(shm, model, ctx_agent_b, "qwen2.5-coder-7b-q4_k_m");

    // ------------------------------------------------------------------------
    // STEP 1: Agent A ingests a large shared document/codebase (e.g., > 64 tokens)
    // ------------------------------------------------------------------------
    std::string shared_context =
        "<|im_start|>system\nYou are a local multi-agent system analyzing a large C++ repository.\n"
        "Here is the shared codebase context that multiple agents will inspect:\n"
        "// [Imagine thousands of tokens of source files here... padding to >64 tokens] "
        "int main() { return 0; } // pad pad pad pad pad pad pad pad pad pad pad pad pad "
        "pad pad pad pad pad pad pad pad pad pad pad pad pad pad pad pad pad pad pad pad\n";

    std::vector<llama_token> shared_tokens = tokenize(model, shared_context, true);

    // Agent A publishes the raw token IDs into SHM and gets a 44-byte TokenRef
    kvarena::TokenRef ref = cache_a.publish_tokens(shared_tokens);

    // Agent A runs its own task by appending its task tokens and calling prefill_with_kv_cache
    std::vector<llama_token> task_a = tokenize(model, "\nTask A: Summarize architecture.<|im_end|>\n", false);
    std::vector<llama_token> prompt_a = shared_tokens;
    prompt_a.insert(prompt_a.end(), task_a.begin(), task_a.end());

    // Computes prefill on GPU/CPU and automatically snapshots the 64-token-aligned KV blocks to SHM
    cache_a.prefill_with_kv_cache(prompt_a, 0);

    // ------------------------------------------------------------------------
    // STEP 2: Agent A sends `ref` (44 bytes) to Agent B over local IPC/pipe
    // ------------------------------------------------------------------------
    // Agent B resolves the shared tokens directly from mmap (Zero-Copy, Zero Tokenization!)
    std::span<const llama_token> zero_copy_prefix = cache_b.resolve_tokens(ref);

    // Agent B appends its own distinct instructions in token space
    std::vector<llama_token> task_b = tokenize(model, "\nTask B: Find memory leaks.<|im_end|>\n", false);
    std::vector<llama_token> prompt_b(zero_copy_prefix.begin(), zero_copy_prefix.end());
    prompt_b.insert(prompt_b.end(), task_b.begin(), task_b.end());

    // Agent B checks deterministic response cache first (Layer C)
    std::vector<llama_token> cached_output;
    uint64_t sampling_hash = 0x12345678ULL; // e.g., temp=0.0, seed=42
    if (!cache_b.lookup_response(prompt_b, sampling_hash, cached_output)) {
        // Cache miss on exact response -> Hydrate KV cache from SHM (Layer A)
        // Agent B skips prefill for all blocks computed by Agent A and ONLY decodes `task_b`!
        cache_b.prefill_with_kv_cache(prompt_b, 0);

        // ... sample tokens using llama_sampler_sample(smpl, ctx_agent_b, -1) ...
        std::vector<llama_token> generated_tokens = { 101, 102, 103 }; // placeholder

        // Cache deterministic response so Agent C doesn't even need to decode
        cache_b.store_response(prompt_b, sampling_hash, generated_tokens);
    }

    llama_free(ctx_agent_b);
    llama_free(ctx_agent_a);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
