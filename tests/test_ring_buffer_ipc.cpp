#include "kvarena.hpp"
#include <cassert>
#include <iostream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

namespace {

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
    const std::string shm_name = "/kvarena_test_rb_" + std::to_string(getpid());
    ShmCleanupGuard cleanup(shm_name);

    constexpr size_t ARENA_BYTES = 16ULL * 1024 * 1024;
    {
        kvarena::SharedArena init_arena(shm_name, ARENA_BYTES, true);
    }

    pid_t pid = fork();
    if (pid == 0) {
        // Child: Consumer
        kvarena::SharedArena arena(shm_name, ARENA_BYTES, false);
        kvarena::TokenRef ref;

        // Busy wait for the token ref
        bool received = false;
        for (int i = 0; i < 1000000; ++i) {
            if (arena.header()->ipc_bus.pop(ref)) {
                received = true;
                break;
            }
            usleep(10);
        }

        assert(received);
        assert(ref.model_hash == 0xDEADBEEF);

        _exit(0);
    }

    // Parent: Producer
    kvarena::SharedArena arena(shm_name, ARENA_BYTES, false);
    kvarena::TokenRef ref;
    ref.model_hash = 0xDEADBEEF;

    bool pushed = false;
    for (int i = 0; i < 1000; ++i) {
        if (arena.header()->ipc_bus.push(ref)) {
            pushed = true;
            break;
        }
        usleep(100);
    }

    assert(pushed);

    int status = 0;
    waitpid(pid, &status, 0);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);

    std::cout << "[PASS] Ring Buffer IPC test completed\n";
    return 0;
}
