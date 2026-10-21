# CMake generated Testfile for
# Source directory: /app
# Build directory: /app/build
#
# This file includes the relevant testing commands required for
# testing this directory and lists subdirectories to be tested as well.
add_test([=[test_kv_radix]=] "/app/build/test_kv_radix")
set_tests_properties([=[test_kv_radix]=] PROPERTIES  TIMEOUT "30" _BACKTRACE_TRIPLES "/app/CMakeLists.txt;48;add_test;/app/CMakeLists.txt;0;")
add_test([=[test_arena_lru]=] "/app/build/test_arena_lru")
set_tests_properties([=[test_arena_lru]=] PROPERTIES  _BACKTRACE_TRIPLES "/app/CMakeLists.txt;58;add_test;/app/CMakeLists.txt;0;")
add_test([=[test_ring_buffer_ipc]=] "/app/build/test_ring_buffer_ipc")
set_tests_properties([=[test_ring_buffer_ipc]=] PROPERTIES  _BACKTRACE_TRIPLES "/app/CMakeLists.txt;62;add_test;/app/CMakeLists.txt;0;")
