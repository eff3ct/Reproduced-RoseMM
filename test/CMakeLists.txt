include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/ff233bdd4cac0a0bf6e5cd45bda3406814cb2796.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(ROSE_MM_TESTS
    rs-estimator-test.cpp
    smcm-ordering-test.cpp
    sub-chain-sparsity-test.cpp
)

add_executable(rose_mm_test ${ROSE_MM_TESTS})

target_link_libraries(
    rose_mm_test
    roseMM
    gtest_main
)

include(GoogleTest)
gtest_discover_tests(rose_mm_test)