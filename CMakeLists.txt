cmake_minimum_required(VERSION 3.20)
project(kvarena VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(KVARENA_BUILD_TESTS "Build libkvarena unit and multi-process tests" ON)
option(KVARENA_USE_MOCK_LLAMA "Use lightweight deterministic mock llama.h for CPU-only CI/Jules VMs" ON)
option(KVARENA_ENABLE_ASAN "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

# Strict compiler warnings as specified in AGENTS.md
add_compile_options(-Wall -Wextra -Werror -Wpedantic)

if(KVARENA_ENABLE_ASAN)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()

find_package(Threads REQUIRED)

# Header-only library target
add_library(kvarena INTERFACE)
target_include_directories(kvarena INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(kvarena INTERFACE Threads::Threads rt)

# Link either the mock llama.cpp stub or a real llama.cpp installation
if(KVARENA_USE_MOCK_LLAMA)
    message(STATUS "libkvarena: Using deterministic mock llama.cpp backend for testing")
    target_include_directories(kvarena INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tests/mock_llama>
    )
else()
    find_package(llama REQUIRED)
    target_link_libraries(kvarena INTERFACE llama)
endif()

# Test Suite Configuration
if(KVARENA_BUILD_TESTS)
    enable_testing()

    add_executable(test_kv_radix tests/test_kv_radix.cpp)
    target_link_libraries(test_kv_radix PRIVATE kvarena)

    add_test(
        NAME test_kv_radix
        COMMAND test_kv_radix
    )
    set_tests_properties(test_kv_radix PROPERTIES TIMEOUT 30)
endif()

if(KVARENA_BUILD_TESTS)
    add_executable(test_arena_lru tests/test_arena_lru.cpp)
    target_link_libraries(test_arena_lru PRIVATE kvarena)
    add_test(NAME test_arena_lru COMMAND test_arena_lru)

    add_executable(test_ring_buffer_ipc tests/test_ring_buffer_ipc.cpp)
    target_link_libraries(test_ring_buffer_ipc PRIVATE kvarena)
    add_test(NAME test_ring_buffer_ipc COMMAND test_ring_buffer_ipc)
endif()
