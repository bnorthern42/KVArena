#include "tokcache.hpp"
#include <cassert>
#include <iostream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

// Helper to tokenize strings via the llama_vocab API
std::vector<llama_token> make_tokens(const llama_model* model, std::string_view text, bool add_bos) {
    const llama_vocab* vocab = llama_model_get_vocab(model);
    int32_t n = -llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()), nullptr, 0, add_bos, true);
    std::vector<llama_token> out(static_cast<size_t>(n));
    llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()), out.data(), n, add_bos, true);
    return out;
}

// RAII Guard to ensure /dev/shm is always cleaned up even if a test fails
struct ShmCleanupGuard {
    std::string name;
    explicit ShmCleanupGuard(std::string n) : name(std::move(n)) {
        shm_unlink(name.c_str());
    }
    ~ShmCleanupGuard() {
        shm_unlink(name.c_str());
    }
};

} // namespace

int main() {
    const std::string shm_name = "/tokcache_test_" + std::to_string(getpid());
    ShmCleanupGuard cleanup(shm_name);

    // Create 16 MB shared memory arena in parent before forking
    constexpr size_t ARENA_BYTES = 16ULL * 1024 * 1024;
    {
        tokcache::SharedArena init_arena(shm_name, ARENA_BYTES, true);
    }

    // Create a pipe solely to pass the 44-byte TokenRef struct from Agent A -> Agent B
    int pipe_fd[2];
    if (pipe(pipe_fd) != 0) {
        std::cerr << "Failed to create IPC pipe\n";
        return 1;
    }

    // Shared prompt > 128 tokens (crosses two 64-token Radix blocks)
    std::string shared_doc(140, 'A');
    std::string suffix_a = "_TASK_A_SUMMARY";
    std::string suffix_b = "_TASK_B_AUDIT";

    // ========================================================================
    // Fork Child 1: Agent A (Publishes shared tokens & populates Radix KV cache)
    // ========================================================================
    pid_t pid_a = fork();
    if (pid_a == 0) {
        close(pipe_fd[0]); // Close read end

        tokcache::SharedArena arena_a(shm_name, ARENA_BYTES, false);
        llama_model model{};
        llama_context ctx_a{};
        ctx_a.model = &model;

        tokcache::AgentTokenCache cache_a(arena_a, &model, &ctx_a, "mock-model-v1");

        auto doc_tokens = make_tokens(&model, shared_doc, true);
        tokcache::TokenRef ref = cache_a.publish_tokens(doc_tokens);

        auto task_a_tokens = make_tokens(&model, suffix_a, false);
        std::vector<llama_token> full_prompt_a = doc_tokens;
        full_prompt_a.insert(full_prompt_a.end(), task_a_tokens.begin(), task_a_tokens.end());


        // Prefill and snapshot 64-token aligned blocks into shared memory
        cache_a.prefill_with_kv_cache(full_prompt_a, 0);

        // Store a deterministic response in Layer C
        std::vector<llama_token> resp_tokens = {9001, 9002, 9003};
        cache_a.store_response(full_prompt_a, 0xCAFEBABEULL, resp_tokens);

        // Send 44-byte TokenRef to Agent B
        if (!arena_a.header()->ipc_bus.push(ref)) {
            _exit(2);
        }
        close(pipe_fd[1]);
        _exit(0);
    }

    int status_a = 0;
    waitpid(pid_a, &status_a, 0);
    assert(WIFEXITED(status_a) && WEXITSTATUS(status_a) == 0);
    close(pipe_fd[1]); // Close write end in parent

    // ========================================================================
    // Fork Child 2: Agent B (Zero-copy resolves TokenRef & hydrates KV prefix)
    // ========================================================================
    pid_t pid_b = fork();
    if (pid_b == 0) {
        tokcache::SharedArena arena_b(shm_name, ARENA_BYTES, false);
        tokcache::TokenRef ref{};
        while (!arena_b.header()->ipc_bus.pop(ref)) { usleep(10); }

        llama_model model{};
        llama_context ctx_b_cached{};
        llama_context ctx_b_baseline{};
        ctx_b_cached.model = &model;
        ctx_b_baseline.model = &model;

        tokcache::AgentTokenCache cache_b(arena_b, &model, &ctx_b_cached, "mock-model-v1");

        // 1. Verify Layer B: Zero-copy token resolution matches Agent A's tokens
        std::span<const llama_token> shared_span = cache_b.resolve_tokens(ref);
        auto expected_doc_tokens = make_tokens(&model, shared_doc, true);
        assert(shared_span.size() == expected_doc_tokens.size());
        assert(std::equal(shared_span.begin(), shared_span.end(), expected_doc_tokens.begin()));

        // 2. Verify Layer C: Deterministic response cache hit for Agent A's query
        auto task_a_tokens = make_tokens(&model, suffix_a, false);
        std::vector<llama_token> full_prompt_a(shared_span.begin(), shared_span.end());
        full_prompt_a.insert(full_prompt_a.end(), task_a_tokens.begin(), task_a_tokens.end());

        std::vector<llama_token> cached_resp;
        bool hit = cache_b.lookup_response(full_prompt_a, 0xCAFEBABEULL, cached_resp);
        assert(hit && cached_resp.size() == 3 && cached_resp[0] == 9001);

        // 3. Verify Layer A: Hydrate 128-token prefix from SHM + decode only suffix_b
        auto task_b_tokens = make_tokens(&model, suffix_b, false);
        std::vector<llama_token> full_prompt_b(shared_span.begin(), shared_span.end());
        full_prompt_b.insert(full_prompt_b.end(), task_b_tokens.begin(), task_b_tokens.end());


        cache_b.prefill_with_kv_cache(full_prompt_b, 0);

        // 4. Compute a full uncached baseline prefill on ctx_b_baseline and compare KV states
        llama_batch batch = llama_batch_init(static_cast<int32_t>(full_prompt_b.size()), 0, 1);
        for (size_t i = 0; i < full_prompt_b.size(); ++i) {
            batch.token[i]     = full_prompt_b[i];
            batch.pos[i]       = static_cast<llama_pos>(i);
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = (i == full_prompt_b.size() - 1);
        }
        batch.n_tokens = static_cast<int32_t>(full_prompt_b.size());
        llama_decode(&ctx_b_baseline, batch);

        // Assert that the hydrated + suffix-decoded KV state is byte-for-byte identical!
        assert(ctx_b_cached.mem.sequences[0].size() == ctx_b_baseline.mem.sequences[0].size());

        for (size_t pos = 0; pos < full_prompt_b.size(); ++pos) {
            const auto& cell_cached   = ctx_b_cached.mem.sequences[0][static_cast<llama_pos>(pos)];
            const auto& cell_baseline = ctx_b_baseline.mem.sequences[0][static_cast<llama_pos>(pos)];
            assert(cell_cached.token == cell_baseline.token);
            assert(cell_cached.pos == cell_baseline.pos);
            assert(cell_cached.rope_rotated_signature == cell_baseline.rope_rotated_signature);
        }

        llama_batch_free(batch);

        _exit(0);
    }

    int status_b = 0;
    waitpid(pid_b, &status_b, 0);
    close(pipe_fd[0]);
    assert(WIFEXITED(status_b) && WEXITSTATUS(status_b) == 0);

    std::cout << "[PASS] Multi-process TokenRef IPC, Response Cache, and Radix KV Hydration verified!\n";
    return 0;
}
