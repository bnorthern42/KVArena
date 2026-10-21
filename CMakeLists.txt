cmake_minimum_required(VERSION 3.20)
project(tokcache LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point to your existing llama.cpp build/source directory
find_package(llama REQUIRED) # Or add_subdirectory(external/llama.cpp)

add_executable(agent_demo agent_demo.cpp)
target_link_libraries(agent_demo PRIVATE llama pthread rt)
