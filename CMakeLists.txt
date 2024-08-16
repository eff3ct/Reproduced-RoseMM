cmake_minimum_required(VERSION 3.24)
project(
    RoseMM
    VERSION 0.0.1
    DESCRIPTION "RoseMM"
    LANGUAGES CXX 
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

include(CTest)

set(SOURCE_DIR src)
set(SOURCES
    ${SOURCE_DIR}/estimator/rs-estimator.cpp
    ${SOURCE_DIR}/estimator/sub-chain-sparsity-estimator.cpp
    ${SOURCE_DIR}/ordering/ordering.cpp
    ${SOURCE_DIR}/multiplication/parallel-matmul.cpp
    ${SOURCE_DIR}/rose-mm/rose-mm.cpp
)

set(HEADER_DIR include)
set(HEADERS
    ${HEADER_DIR}/common/matrix.hpp
    ${HEADER_DIR}/ordering.hpp
    ${HEADER_DIR}/rs-estimator.hpp
    ${HEADER_DIR}/rose-mm.hpp
    ${HEADER_DIR}/sub-chain-sparsity-estimator.hpp
    ${HEADER_DIR}/parallel-matmul.hpp
)

add_library(roseMM STATIC ${HEADERS} ${SOURCES})

add_subdirectory(test)

target_include_directories(roseMM
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
)