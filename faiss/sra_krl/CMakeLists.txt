cmake_minimum_required(VERSION 3.0)

project(krl)

add_compile_options(-O3 -march=armv8.2-a+fp16fml+dotprod -fPIC -fvisibility=hidden -fstack-protector-strong -D_FORTIFY_SOURCE=2 -ftrapv -std=c++17)
add_link_options(-O3 -march=armv8.2-a+fp16fml+dotprod -fPIC -fvisibility=hidden -Wl,-z,relro,-z,now -s -std=c++17)

set_property(SOURCE ${CMAKE_SOURCE_DIR}/src/sve_search_codes.c APPEND PROPERTY COMPILE_FLAGS "-march=armv8.2-a+sve -msve-vector-bits=256")

set(CMAKE_C_COMPILER "g++")

aux_source_directory(${CMAKE_SOURCE_DIR}/src SRC_LIST)

include_directories(${CMAKE_SOURCE_DIR}/include)

set(KRL_VERSION_MAJOR 1)
set(KRL_VERSION_MINOR 0)
set(KRL_VERSION_PATCH 0)

add_library(func_shared SHARED ${SRC_LIST})

set_target_properties(func_shared PROPERTIES 
    OUTPUT_NAME "krl"
        VERSION "${KRL_VERSION_MAJOR}.${KRL_VERSION_MINOR}.${KRL_VERSION_PATCH}"
            SOVERSION "${KRL_VERSION_MAJOR}"
                LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
)