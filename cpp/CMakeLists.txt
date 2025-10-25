cmake_minimum_required(VERSION 3.20)
project(Jakube)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Build Options ---

# Add an option to control the multithreaded build
# This MUST be defined before add_subdirectory(test)
option(BUILD_MULTITHREADED "Enable multithreaded index building" ON)

if(BUILD_MULTITHREADED)
    message(STATUS "Multithreaded build policy is ON")
    # This adds the -DJAKUBELIB_MULTITHREADED_BUILD flag
    # for all targets in this project and sub-projects
    add_compile_definitions(JAKUBELIB_MULTITHREADED_BUILD)
endif()

# --- Subdirectories ---

enable_testing()

add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(test)

# --- Formatting Targets ---

# Find all source and header files in the subdirectories
# This is more portable than using the 'find' command
file(GLOB_RECURSE FORMAT_FILES
    "src/*.cpp"
    "src/*.h"
    "include/*.h"
    "test/*.cpp"
    "test/*.h"
)

# Define the clang-format commands
set(CHECK_FORMAT_COMMAND clang-format --verbose --dry-run -i)
set(FORMAT_COMMAND clang-format --verbose -i)

# Check formatting only
add_custom_target(check-formatting
    COMMAND ${CHECK_FORMAT_COMMAND} ${FORMAT_FILES}
    COMMENT "Checking C++ formatting"
)

# Run formatter
add_custom_target(format
    COMMAND ${FORMAT_COMMAND} ${FORMAT_FILES}
    COMMENT "Running C++ formatter"
)
