# AGENTS.md — `libkvarena` Agent Guide

## Project Overview
`libkvarena` is a C++20 library that enables multiple local AI agent processes running `llama.cpp` to share:
1. **Layer A (Radix KV-Cache):** Precomputed `llama.cpp` sequence KV states indexed by chained 64-token block hashes.
2. **Layer B (Zero-Copy Token IPC):** Raw `llama_token` (`int32_t`) arrays in POSIX shared memory referenced via lightweight `TokenRef` descriptors.
3. **Layer C (Deterministic Response Cache):** Exact output token sequences keyed by `H(model_hash ^ sampling_hash, prompt_tokens)`.

---

## Critical Architectural Invariants (DO NOT VIOLATE)

1. **NO Raw Pointers in Shared Memory (`mmap`):**
   - Multiple OS processes map the shared memory segment at different virtual base addresses.
   - Every struct stored inside `ShmHeader` or the shared data slab (`RadixBlockEntry`, `ResponseCacheEntry`, allocator headers, ring buffer nodes) **MUST** use `uint64_t` byte offsets relative to `slab_base_` (with `0` reserved as the null offset).
   - Never store `std::vector`, `std::string`, `void*`, or polymorphic types (vtable pointers) inside shared memory structs. All shared structs must satisfy `std::is_trivially_copyable_v<T>` and `std::is_standard_layout_v<T>` (`static_assert` this for every shared struct).

2. **Cross-Process Atomics & Synchronization:**
   - When performing atomic operations on variables inside the `mmap` region, use `std::atomic_ref<T>` (C++20) where `std::atomic_ref<T>::is_always_lock_free` is verified via `static_assert`, and ensure 8-byte or 64-byte alignment (`alignas(64)` for ring buffer head/tail indices to prevent false sharing).
   - Process-shared locks must be initialized with `PTHREAD_PROCESS_SHARED`.

3. **Alignment Requirements:**
   - `llama_token` buffers in the shared slab must be aligned to `alignof(llama_token)` (4 bytes).
   - Serialized `llama.cpp` KV state buffers must be aligned to **64 bytes** (`alignas(64)`) to satisfy SIMD/AVX/GGML tensor read alignment when calling `llama_state_seq_set_data` / `llama_state_seq_get_data`.

4. **Current `llama.cpp` API Conventions:**
   - Use `const llama_vocab* vocab = llama_model_get_vocab(model);` and `llama_tokenize(vocab, ...)` (do not use deprecated model-direct tokenize signatures).
   - Use `llama_state_seq_get_size(ctx, seq_id)`, `llama_state_seq_get_data(ctx, dst, size, seq_id)`, and `llama_state_seq_set_data(ctx, src, size, seq_id)` (or their `*_ext` variants).
   - For RoPE position shifts on cached sequences, use `llama_memory_seq_add(llama_get_memory(ctx), seq_id, p0, p1, delta)`.

---

## Repository Structure
```text
.
├── AGENTS.md
├── ROADMAP.md
├── CMakeLists.txt
├── include/
│   ├── kvarena.hpp              # Umbrella header
│   └── kvarena/
│       ├── arena.hpp             # SharedArena, ShmHeader, LRU slab allocator
│       ├── ring_buffer.hpp       # Lock-free MPMC TokenRef IPC bus
│       ├── radix_kv.hpp          # Chained block hashing & KV checkpointing
│       ├── response_cache.hpp    # Deterministic prompt->output token cache
│       └── client.hpp            # AgentTokenCache (unified llama.cpp adapter)
├── examples/
│   └── agent_demo.cpp            # Two-agent shared prefill & IPC demo
└── tests/
    ├── test_arena_lru.cpp        # Slab allocation, pinning & LRU eviction tests
    ├── test_ring_buffer_ipc.cpp  # Multi-process fork() MPMC ring buffer tests
    └── test_kv_radix.cpp         # Radix block matching & llama.cpp state tests

Build & Test Commands

When running in the Jules VM environment, build and run tests using CMake and CTest:
Bash

# Configure (build tests even if a full GGUF model isn't present via mock KV mode)
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DKVARENA_BUILD_TESTS=ON -DKVARENA_ENABLE_ASAN=ON

# Build all targets
cmake --build build -j$(nproc)

# Run unit and multi-process integration tests
ctest --test-dir build --output-on-failure

Shared Memory Cleanup During Testing

    Tests must use unique POSIX shared memory names (e.g., "/kvarena_test_" + std::to_string(getpid())) and explicitly call shm_unlink() in test teardown so /dev/shm does not leak segments across test runs.

Code Style Guidelines

    Standard: C++20 (-std=c++20 -Wall -Wextra -Werror -Wpedantic).

    Headers: Header-only core library under namespace kvarena for zero-overhead inlining, with optional compiled tools/tests.

    Error Handling: Prefer std::expected-style return codes or bool status flags on hot cache lookup paths; reserve exceptions (std::runtime_error) strictly for unrecoverable initialization or mmap setup failures.


---

### Quick Tip for Jules Setup
In the Jules repo settings, you can also add a one-line **Environment Setup Script** if `llama.cpp` isn't vendored as a git submodule yet:
```bash
git submodule update --init --recursive && cmake -B build -DKVARENA_BUILD_TESTS=ON && cmake --build build -j$(nproc)
