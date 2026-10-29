#pragma once

#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <pthread.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <atomic>

namespace kvarena {

constexpr size_t BLOCK_TOKENS = 64;          // Granularity for prefix matching
constexpr size_t MAX_INDEX_ENTRIES = 4096;   // Fixed-size table in SHM
constexpr size_t MAX_RESP_ENTRIES = 1024;    // Deterministic response cache entries
constexpr uint64_t MAGIC_HEADER = 0x544F4B4341434845ULL; // "KVARENA"

// Forward declare TokenRef for ring buffer
struct TokenRef;

// Forward declare ring buffer channel to use in ShmHeader
template <typename T, size_t Capacity> struct RingBufferChannel;

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

inline uint64_t get_monotonic_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + ts.tv_nsec;
}

struct RadixBlockEntry {
    uint64_t block_hash;      // Chained hash: H(parent_hash + model_hash + 64 tokens)
    uint64_t parent_hash;     // Hash of preceding block (0 for root)
    uint64_t token_offset;    // Offset in slab where full prefix tokens are stored
    uint32_t total_tokens;    // Total tokens from pos 0 up to the end of this block
    uint64_t kv_offset;       // Offset in slab where llama_state_seq_get_data is stored
    uint64_t kv_size;         // Byte size of the KV state snapshot
    uint64_t last_access_ts;  // For LRU eviction
    // Robin Hood / Tombstone fields added in Step 5
    uint32_t probe_dist;
    bool is_valid;
    bool is_tombstone;
};

struct ResponseCacheEntry {
    uint64_t query_hash;      // H(model_hash + full_prompt_tokens + sampling_params)
    uint64_t token_offset;    // Offset in slab to generated output llama_token[]
    uint32_t n_output_tokens; // Length of cached output token sequence
    // Robin Hood / Tombstone fields added in Step 5
    uint32_t probe_dist;
    bool is_valid;
    bool is_tombstone;
};

// ============================================================================
// Slab Allocation and LRU Intrusive List
// ============================================================================
struct AllocHeader {
    uint64_t size;            // Size of the block including header
    uint64_t ref_count;       // Atomic reference count for pinning
    uint64_t last_access_ts;  // LRU timestamp
    uint64_t prev_lru;        // Offset to previous block in LRU list (0 = null)
    uint64_t next_lru;        // Offset to next block in LRU list (0 = null)
    uint64_t prev_free;       // Offset to previous block in free list (0 = null)
    uint64_t next_free;       // Offset to next block in free list (0 = null)
    bool is_free;             // Whether this block is free
};

// ============================================================================
// Lock-Free MPMC Shared Memory Ring Buffer
// ============================================================================
template <typename T, size_t Capacity>
struct RingBufferChannel {
    static_assert((Capacity != 0) && ((Capacity & (Capacity - 1)) == 0), "Capacity must be power of 2");

    alignas(64) uint64_t raw_head;
    alignas(64) uint64_t raw_tail;

    struct RawNode {
        uint64_t sequence;
        T data;
    };
    RawNode nodes[Capacity];

    void init() {
        raw_head = 0;
        raw_tail = 0;
        for (size_t i = 0; i < Capacity; ++i) {
            nodes[i].sequence = i;
        }
    }

    bool push(const T& data) {
        std::atomic_ref<uint64_t> head_ref(raw_head);

        while (true) {
            uint64_t pos = head_ref.load(std::memory_order_relaxed);
            size_t idx = pos & (Capacity - 1);

            std::atomic_ref<uint64_t> seq_ref(nodes[idx].sequence);
            uint64_t seq = seq_ref.load(std::memory_order_acquire);

            intptr_t dif = (intptr_t)seq - (intptr_t)pos;

            if (dif == 0) {
                if (head_ref.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    nodes[idx].data = data;
                    seq_ref.store(pos + 1, std::memory_order_release);
                    return true;
                }
            } else if (dif < 0) {
                return false; // Queue is full
            }
        }
    }

    bool pop(T& data) {
        std::atomic_ref<uint64_t> tail_ref(raw_tail);

        while (true) {
            uint64_t pos = tail_ref.load(std::memory_order_relaxed);
            size_t idx = pos & (Capacity - 1);

            std::atomic_ref<uint64_t> seq_ref(nodes[idx].sequence);
            uint64_t seq = seq_ref.load(std::memory_order_acquire);

            intptr_t dif = (intptr_t)seq - (intptr_t)(pos + 1);

            if (dif == 0) {
                if (tail_ref.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    data = nodes[idx].data;
                    seq_ref.store(pos + Capacity, std::memory_order_release);
                    return true;
                }
            } else if (dif < 0) {
                return false; // Queue is empty
            }
        }
    }
};

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
    uint64_t token_header_offset; // added to access header for pinning
    uint64_t kv_header_offset;
};

struct ShmHeader {
    uint64_t magic;
    pthread_rwlock_t rwlock;  // Cross-process read-write lock
    uint64_t slab_capacity;

    uint64_t lru_head;        // Offset to head of LRU list (most recently used)
    uint64_t lru_tail;        // Offset to tail of LRU list (least recently used)
    uint64_t free_head;       // Offset to head of free list

    RadixBlockEntry radix_table[MAX_INDEX_ENTRIES];
    ResponseCacheEntry resp_table[MAX_RESP_ENTRIES];
    RingBufferChannel<TokenRef, 256> ipc_bus; // RingBuffer added here
};

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

            // Initialize free list with one large block starting at offset 64 (to reserve 0 as null)
            uint64_t initial_offset = 64;
            header_->free_head = initial_offset;
            header_->lru_head = 0;
            header_->lru_tail = 0;

            header_->ipc_bus.init(); // Init ring buffer

            header_->magic = MAGIC_HEADER;

            AllocHeader* first_block = ptr_from_offset<AllocHeader>(initial_offset);
            std::memset(first_block, 0, sizeof(AllocHeader));
            first_block->size = header_->slab_capacity - initial_offset;
            first_block->is_free = true;
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

    // LRU and Free list management
    void lru_remove(uint64_t offset) {
        AllocHeader* hdr = ptr_from_offset<AllocHeader>(offset);
        if (hdr->prev_lru != 0) {
            ptr_from_offset<AllocHeader>(hdr->prev_lru)->next_lru = hdr->next_lru;
        } else {
            header_->lru_head = hdr->next_lru;
        }
        if (hdr->next_lru != 0) {
            ptr_from_offset<AllocHeader>(hdr->next_lru)->prev_lru = hdr->prev_lru;
        } else {
            header_->lru_tail = hdr->prev_lru;
        }
        hdr->prev_lru = 0;
        hdr->next_lru = 0;
    }

    void lru_push_front(uint64_t offset) {
        AllocHeader* hdr = ptr_from_offset<AllocHeader>(offset);
        hdr->prev_lru = 0;
        hdr->next_lru = header_->lru_head;
        if (header_->lru_head != 0) {
            ptr_from_offset<AllocHeader>(header_->lru_head)->prev_lru = offset;
        } else {
            header_->lru_tail = offset;
        }
        header_->lru_head = offset;
    }

    void free_remove(uint64_t offset) {
        AllocHeader* hdr = ptr_from_offset<AllocHeader>(offset);
        if (hdr->prev_free != 0) {
            ptr_from_offset<AllocHeader>(hdr->prev_free)->next_free = hdr->next_free;
        } else {
            header_->free_head = hdr->next_free;
        }
        if (hdr->next_free != 0) {
            ptr_from_offset<AllocHeader>(hdr->next_free)->prev_free = hdr->prev_free;
        }
        hdr->prev_free = 0;
        hdr->next_free = 0;
    }

    void free_push_front(uint64_t offset) {
        AllocHeader* hdr = ptr_from_offset<AllocHeader>(offset);
        hdr->prev_free = 0;
        hdr->next_free = header_->free_head;
        if (header_->free_head != 0) {
            ptr_from_offset<AllocHeader>(header_->free_head)->prev_free = offset;
        }
        header_->free_head = offset;
    }

    void evict_from_tables(uint64_t data_offset) {
        // Linear scan for now, will be optimized if needed, but Robin Hood will help.
        // Data offset is offset of user data
        for (size_t i = 0; i < MAX_INDEX_ENTRIES; ++i) {
            if (header_->radix_table[i].is_valid && !header_->radix_table[i].is_tombstone) {
                if (header_->radix_table[i].token_offset == data_offset ||
                    header_->radix_table[i].kv_offset == data_offset) {
                    header_->radix_table[i].is_tombstone = true;
                    header_->radix_table[i].is_valid = false;
                }
            }
        }
        for (size_t i = 0; i < MAX_RESP_ENTRIES; ++i) {
            if (header_->resp_table[i].is_valid && !header_->resp_table[i].is_tombstone) {
                if (header_->resp_table[i].token_offset == data_offset) {
                    header_->resp_table[i].is_tombstone = true;
                    header_->resp_table[i].is_valid = false;
                }
            }
        }
    }

    // Allocate aligned bytes in the shared slab (caller must hold write lock)
    uint64_t allocate(size_t bytes, size_t alignment = 8) {
        // Need space for user data + AllocHeader + padding for alignment
        // Actual block size depends on where it is aligned. Let's simplify and ensure the block start is aligned so data is aligned.

        // Ensure AllocHeader size is a multiple of maximum expected alignment (64) to make things easier, or just calculate exactly.
        size_t header_size = (sizeof(AllocHeader) + 63) & ~63;
        size_t total_needed = header_size + bytes;

        uint64_t curr_free = header_->free_head;
        uint64_t best_free = 0;

        while (curr_free != 0) {
            AllocHeader* hdr = ptr_from_offset<AllocHeader>(curr_free);
            // Calculate alignment padding
            uint64_t data_start = curr_free + header_size;
            uint64_t aligned_data_start = (data_start + (alignment - 1)) & ~(alignment - 1);
            size_t padding = aligned_data_start - data_start;

            if (hdr->size >= total_needed + padding) {
                best_free = curr_free;
                break; // First fit
            }
            curr_free = hdr->next_free;
        }

        if (best_free == 0) {
            // Need to evict LRU unpinned entries
            uint64_t curr_lru = header_->lru_tail;
            bool freed_something = false;
            while (curr_lru != 0) {
                AllocHeader* hdr = ptr_from_offset<AllocHeader>(curr_lru);
                uint64_t prev = hdr->prev_lru;

                std::atomic_ref<uint64_t> ref_count(hdr->ref_count);
                if (ref_count.load() == 0) {
                    // Evict
                    lru_remove(curr_lru);
                    evict_from_tables(curr_lru + header_size); // Approx data offset for table clearing

                    hdr->is_free = true;
                    free_push_front(curr_lru);

                    // Simple coalescing with adjacent physically?
                    // We need physical next/prev for proper coalescing, which we don't have.
                    // For now, let's assume simple eviction gives enough space, or implement a naive scan to coalesce.
                    // To do proper coalescing, we'd need physical block size iteration.
                    freed_something = true;
                }
                curr_lru = prev;
            }

            // Coalesce all free blocks (naive O(N) sweep)
            uint64_t physical_curr = 64; // initial offset
            while (physical_curr < header_->slab_capacity) {
                AllocHeader* phys_hdr = ptr_from_offset<AllocHeader>(physical_curr);
                if (phys_hdr->size == 0) break; // Corrupted?

                if (phys_hdr->is_free) {
                    uint64_t next_phys = physical_curr + phys_hdr->size;
                    while (next_phys < header_->slab_capacity) {
                        AllocHeader* next_hdr = ptr_from_offset<AllocHeader>(next_phys);
                        if (next_hdr->is_free) {
                            // Coalesce
                            free_remove(next_phys);
                            phys_hdr->size += next_hdr->size;
                            next_phys = physical_curr + phys_hdr->size;
                        } else {
                            break;
                        }
                    }
                }
                physical_curr += phys_hdr->size;
            }

            if (!freed_something) {
                throw std::runtime_error("OOM: Shared arena full and no unpinned entries to evict.");
            }

            // Try allocate again
            return allocate(bytes, alignment);
        }

        // Allocate from best_free
        AllocHeader* hdr = ptr_from_offset<AllocHeader>(best_free);
        uint64_t data_start = best_free + header_size;
        uint64_t aligned_data_start = (data_start + (alignment - 1)) & ~(alignment - 1);
        size_t padding = aligned_data_start - data_start;

        uint64_t needed_size = total_needed + padding;

        if (hdr->size > needed_size + header_size + 64) { // Minimum split size
            // Split
            uint64_t new_free_offset = best_free + needed_size;
            AllocHeader* new_free_hdr = ptr_from_offset<AllocHeader>(new_free_offset);
            std::memset(new_free_hdr, 0, sizeof(AllocHeader));
            new_free_hdr->size = hdr->size - needed_size;
            new_free_hdr->is_free = true;

            hdr->size = needed_size;

            free_remove(best_free);
            free_push_front(new_free_offset);
        } else {
            free_remove(best_free);
        }

        hdr->is_free = false;
        hdr->ref_count = 0;
        hdr->last_access_ts = get_monotonic_ns();
        lru_push_front(best_free);

        return aligned_data_start;
    }

    void touch(uint64_t offset) {
        if (offset == 0) return;
        // Data offset to header offset: we don't know exact header offset due to padding.
        // Actually, let's keep it simple: we can walk physical blocks to find it, or store header offset before data.
        // Let's store header offset right before the aligned data start.
    }

    template <typename T>
    T* ptr_from_offset(uint64_t offset) const {
        if (offset == 0) return nullptr;
        return reinterpret_cast<T*>(slab_base_ + offset);
    }

    ShmHeader* header() const { return header_; }

    uint8_t* slab_base() const { return slab_base_; }

private:
    std::string shm_name_;
    size_t total_bytes_;
    int fd_ = -1;
    uint8_t* base_ = nullptr;
    ShmHeader* header_ = nullptr;
    uint8_t* slab_base_ = nullptr;
};

} // namespace kvarena
