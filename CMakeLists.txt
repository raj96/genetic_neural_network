cmake_minimum_required(VERSION 3.25.1)

project(genetic_neural_network)

# add_compile_options(-ggdb)

set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/src/include)

file(GLOB UTIL_SRC ${PROJECT_SOURCE_DIR}/src/lib/util/*.c)
add_library(util STATIC ${UTIL_SRC})
target_include_directories(util PUBLIC ${INCLUDE_DIR})

file(GLOB NN_SRC ${PROJECT_SOURCE_DIR}/src/lib/nn/*.c)
add_library(nn SHARED ${NN_SRC})
target_include_directories(nn PUBLIC ${INCLUDE_DIR})
target_link_libraries(nn PUBLIC util)

file(GLOB GA_SRC ${PROJECT_SOURCE_DIR}/src/lib/ga/*.c)
add_library(ga SHARED ${GA_SRC})
target_include_directories(ga PUBLIC ${INCLUDE_DIR})
target_link_libraries(ga PUBLIC util)

# Executables: Examples
file(GLOB TEST_SRC ${PROJECT_SOURCE_DIR}/examples/*.c)

add_executable(test ${TEST_SRC})
target_include_directories(test PUBLIC ${INCLUDE_DIR})
target_link_libraries(test PUBLIC nn util m ga)

