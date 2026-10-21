# `libtokcache` Roadmap

**Goal:** A high-performance, zero-copy C++20 library enabling local LLM agents backed by `llama.cpp` to share tokenized contexts, Radix-indexed KV-cache states, and deterministic responses over POSIX shared memory (`mmap`).

---

## Phase 1: Core Prototype & Three-Layer Foundation (Completed)
- [x] **Shared Memory Arena (`SharedArena`)**: Cross-process POSIX `shm_open` + `mmap` segment protected by `PTHREAD_PROCESS_SHARED` read-write locks.
- [x] **Layer B — Zero-Copy Token IPC (`TokenRef`)**: 44-byte offset-based descriptor allowing agents to publish and resolve `std::span<const llama_token>` directly from shared memory without detokenizing or copying strings.
- [x] **Layer A — Chained Block-Hash KV Cache**: 64-token rolling FNV-1a block hashing (`RadixBlockEntry`) paired with `llama_state_seq_get_data` and `llama_state_seq_set_data` for prefix hydration and suffix-only `llama_decode`.
- [x] **Layer C — Deterministic Response Cache**: Prompt + sampling hash table (`ResponseCacheEntry`) for zero-compute retrieval of repeated sub-agent queries.

---

## Phase 2: Memory Management, Collision Handling & Lock-Free IPC (Current Target)
- [ ] **Modular Header Layout**:
  - Split `include/tokcache.hpp` into `include/tokcache/{arena,ring_buffer,radix_kv,response_cache,client}.hpp`.
- [ ] **LRU Slab Allocator & Pinning**:
  - Replace the naive wrap-around bump allocator with a block/slab allocator.
  - Add atomic `ref_count` pinning to `RadixBlockEntry` and `TokenRef` allocations so an agent reading a shared span cannot have its memory evicted by a writer.
  - Implement monotonic timestamp updates (`clock_gettime(CLOCK_MONOTONIC, ...)`) on cache hits and evict least-recently-used unpinned entries when the slab fills.
- [ ] **Robin Hood Open-Addressing Hash Tables**:
  - Upgrade `radix_table` and `resp_table` from direct-mapped slots to Robin Hood linear probing with probe-distance tracking and safe deletion tombstones.
- [ ] **Lock-Free SHM Ring Buffer (`RingBufferChannel`)**:
  - Embed a cacheline-aligned (`alignas(64)`) MPMC ring buffer in `ShmHeader` using `std::atomic_ref<uint64_t>` on shared memory indices so agents can pub/sub `TokenRef` envelopes with sub-microsecond latency.
- [ ] **Multi-Process Test Suite**:
  - Add `fork()`-based multi-process stress tests verifying concurrent read/write safety, ring-buffer pub/sub, and LRU eviction under contention.

---

## Phase 3: Advanced `llama.cpp` KV Manipulation (Delta Snapshots & RoPE Shifting)
- [ ] **Incremental / Cell-Range KV Extraction**:
  - Avoid storing redundant prefix bytes at every 64-token checkpoint by leveraging `llama_state_seq_get_data_ext` / `llama_state_seq_set_data_ext` (or coarse checkpoint intervals at 256/512 tokens combined with delta blocks).
- [ ] **Non-Prefix Chunk Reuse via RoPE Position Shifting**:
  - Allow Agent B to reuse a cached document chunk at token positions `[p0' .. p1']` that Agent A originally computed at `[p0 .. p1]`.
  - Integrate `llama_memory_seq_add(llama_get_memory(ctx), seq_id, p0, p1, delta)` after hydrating a chunk to rotate its baked-in Rotary Position Embeddings (RoPE) in-place.
- [ ] **Robust Process Crash Recovery**:
  - Configure `pthread_mutexattr_setrobust(&attr, PTHREAD_MUTEX_ROBUST)` / robust lock recovery so a crashed agent process holding a lock does not deadlock the entire shared arena.

---

## Phase 4: Observability, C ABI & Multi-Language Bindings
- [ ] **Shared Telemetry Counters**:
  - Track `prefill_tokens_saved`, `kv_bytes_hydrated`, `resp_cache_hits`, `slab_evictions`, and `lock_contention_ns` via atomic counters in `ShmHeader`.
  - Build a lightweight CLI inspector (`tokcache-top`) that attaches to `/tokcache_bus` read-only and displays live cache utilization and agent hit rates.
- [ ] **Stable C ABI (`tokcache_c.h`)**:
  - Expose an `extern "C"` wrapper so Python, Rust, Go, or Elixir agent orchestrators can pass `TokenRef` handles to C++ `llama.cpp` worker processes.
