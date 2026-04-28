cmake_minimum_required(VERSION 3.0)

project(krl)

option(USE_SVE2 "Enable SVE2 kernels and public API symbols" OFF)

add_compile_options(
    -O3
    -march=armv8.2-a+fp16fml+dotprod
    -fPIC
    -fvisibility=hidden
    -fstack-protector-strong
    -D_FORTIFY_SOURCE=2
    -ftrapv
    -std=c++17
)
add_link_options(
    -O3
    -march=armv8.2-a+fp16fml+dotprod
    -fPIC
    -fvisibility=hidden
    -Wl,-z,relro,-z,now
    -s
    -std=c++17
)

set(CMAKE_C_COMPILER "g++")

set(KRL_BASE_SRC
    ${CMAKE_SOURCE_DIR}/src/handle_IO.c
    ${CMAKE_SOURCE_DIR}/src/Heap_sort.c
    ${CMAKE_SOURCE_DIR}/src/IPdistance_simd_f16.c
    ${CMAKE_SOURCE_DIR}/src/IPdistance_simd_f16f32.c
    ${CMAKE_SOURCE_DIR}/src/IPdistance_simd_s8.c
    ${CMAKE_SOURCE_DIR}/src/IPdistance_simd.c
    ${CMAKE_SOURCE_DIR}/src/krl_handles.c
    ${CMAKE_SOURCE_DIR}/src/L2distance_simd_f16.c
    ${CMAKE_SOURCE_DIR}/src/L2distance_simd_f16f32.c
    ${CMAKE_SOURCE_DIR}/src/L2distance_simd_u8.c
    ${CMAKE_SOURCE_DIR}/src/L2distance_simd.c
    ${CMAKE_SOURCE_DIR}/src/matrix_block_transpose.c
    ${CMAKE_SOURCE_DIR}/src/MinMax_quant.c
    ${CMAKE_SOURCE_DIR}/src/NegaIPdistance_simd_f16f32.c
    ${CMAKE_SOURCE_DIR}/src/NegaIPdistance_simd_s8.c
    ${CMAKE_SOURCE_DIR}/src/pq_search_with_table_4bit.c
    ${CMAKE_SOURCE_DIR}/src/pq_search_with_table_8bit.c
    ${CMAKE_SOURCE_DIR}/src/reorder_2_vectors.c
    ${CMAKE_SOURCE_DIR}/src/sve_search_codes.c
)

set(KRL_SVE2_SRC
    ${CMAKE_SOURCE_DIR}/src/L2distance_simd_f16f32_sve2.c
    ${CMAKE_SOURCE_DIR}/src/IPdistance_simd_f16f32_sve2.c
    ${CMAKE_SOURCE_DIR}/src/pq_search_with_table_4bit_sve2.c
)

set(SRC_LIST ${KRL_BASE_SRC})
if(USE_SVE2)
    list(APPEND SRC_LIST ${KRL_SVE2_SRC})
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

set(KRL_VERSION_MAJOR 1)
set(KRL_VERSION_MINOR 0)
set(KRL_VERSION_PATCH 0)

add_library(func_shared SHARED ${SRC_LIST})

set_source_files_properties(${SRC_LIST} PROPERTIES LANGUAGE CXX)
set_property(
    SOURCE ${CMAKE_SOURCE_DIR}/src/sve_search_codes.c
    APPEND PROPERTY COMPILE_FLAGS " -march=armv8.2-a+sve -msve-vector-bits=256"
)

if(USE_SVE2)
    target_compile_definitions(func_shared PRIVATE USE_SVE2)
    set_property(
        SOURCE ${KRL_SVE2_SRC}
        APPEND PROPERTY COMPILE_FLAGS " -march=armv8.2-a+fp16+sve2+rcpc -msve-vector-bits=256"
    )
endif()

set_target_properties(func_shared PROPERTIES 
    OUTPUT_NAME "krl"
        VERSION "${KRL_VERSION_MAJOR}.${KRL_VERSION_MINOR}.${KRL_VERSION_PATCH}"
            SOVERSION "${KRL_VERSION_MAJOR}"
                LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
)