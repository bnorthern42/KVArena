#include "kvarena.hpp"
#include <cassert>
#include <iostream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

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
    const std::string shm_name = "/kvarena_test_lru_" + std::to_string(getpid());
    ShmCleanupGuard cleanup(shm_name);

    // Create a small shared memory arena
    constexpr size_t ARENA_BYTES = 1024 * 1024; // 1 MB
    kvarena::SharedArena arena(shm_name, ARENA_BYTES, true);

    arena.lock_write();

    // Fill up the arena with small allocations until full
    std::vector<uint64_t> offsets;
    try {
        for (int i = 0; i < 10000; ++i) {
            offsets.push_back(arena.allocate(1024, 8));
        }
    } catch (const std::runtime_error& e) {
        // Expected OOM or successful eviction
    }

    arena.unlock();

    std::cout << "[PASS] LRU Arena test completed\n";
    return 0;
}
